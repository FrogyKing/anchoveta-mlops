import os
from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics

# Use the same image in training and inference so joblib can deserialize
# CatBoost / LightGBM / XGBoost models with compatible library versions.
IMAGE_URI = "us-central1-docker.pkg.dev/anchoveta/mlops-repo/anchoveta-pipeline:latest"
print(f"DEBUG: Compiling with IMAGE_URI={IMAGE_URI}")


@dsl.component(base_image=IMAGE_URI)
def extract_data_from_bq(
    project_id: str,
    bq_table: str,
    dataset_out: Output[Dataset],
):
    from google.cloud import bigquery
    from google.cloud import bigquery_storage
    import pandas as pd

    client = bigquery.Client(project=project_id)
    bqstorage_client = bigquery_storage.BigQueryReadClient()

    query = f"SELECT * FROM `{bq_table}`"
    df = (
        client.query(query)
        .result()
        .to_dataframe(
            bqstorage_client=bqstorage_client,
            create_bqstorage_client=False,
        )
    )

    # BigQuery Storage API can return dbdate/dbtime objects that PyArrow
    # cannot always serialize directly. Feature engineering must restore
    # date/numeric types where required.
    for col in df.select_dtypes(include=["dbdate", "object"]).columns:
        df[col] = df[col].astype(str)

    df.to_parquet(dataset_out.path, index=False)


@dsl.component(base_image=IMAGE_URI)
def train_catboost(
    dataset_in: Input[Dataset],
    model_artifact: Output[Model],
    metrics_out: Output[Metrics],
    n_trials: int = 30,
):
    import json
    import joblib
    import numpy as np
    import optuna
    import pandas as pd
    from catboost import CatBoostRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from src.utils.features import apply_feature_engineering, FEATURES, TARGET

    def temporal_split(data: pd.DataFrame):
        data = data.copy()
        data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")
        data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
        data = (
            data.dropna(subset=["fecha", TARGET])
            .sort_values("fecha")
            .reset_index(drop=True)
        )

        unique_dates = np.sort(data["fecha"].unique())
        if len(unique_dates) < 3:
            raise ValueError("Se necesitan al menos 3 fechas distintas para train/validación/test.")

        train_pos = max(0, min(len(unique_dates) - 3, int(len(unique_dates) * 0.70) - 1))
        val_pos = max(train_pos + 1, min(len(unique_dates) - 2, int(len(unique_dates) * 0.85) - 1))

        train_end = unique_dates[train_pos]
        val_end = unique_dates[val_pos]

        train = data[data["fecha"] <= train_end]
        valid = data[(data["fecha"] > train_end) & (data["fecha"] <= val_end)]
        test = data[data["fecha"] > val_end]

        if train.empty or valid.empty or test.empty:
            raise ValueError(
                f"Split temporal inválido: train={len(train)}, valid={len(valid)}, test={len(test)}"
            )
        return train, valid, test

    def evaluate(y_true, pred_log):
        rmse = float(np.sqrt(mean_squared_error(y_true, pred_log)))
        mae_log = float(mean_absolute_error(y_true, pred_log))
        r2_log = float(r2_score(y_true, pred_log))
        mae_real = float(mean_absolute_error(np.expm1(y_true), np.expm1(pred_log)))
        return rmse, mae_log, r2_log, mae_real

    df = pd.read_parquet(dataset_in.path)
    df = apply_feature_engineering(df)
    train_df, valid_df, test_df = temporal_split(df)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_valid, y_valid = valid_df[FEATURES], valid_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": 4000,
            "depth": trial.suggest_int("depth", 4, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1e-2, 50.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.0, 5.0),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 5.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "loss_function": "RMSE",
            "eval_metric": "RMSE",
            "random_seed": 42,
            "verbose": 0,
            "allow_writing_files": False,
            "thread_count": -1,
        }

        candidate = CatBoostRegressor(**params)
        candidate.fit(
            X_train,
            y_train,
            eval_set=(X_valid, y_valid),
            early_stopping_rounds=150,
            use_best_model=True,
            verbose=False,
        )
        pred = candidate.predict(X_valid)
        best_iteration = candidate.get_best_iteration()
        trial.set_user_attr(
            "best_iteration",
            int(best_iteration + 1 if best_iteration is not None and best_iteration >= 0 else params["iterations"]),
        )
        return float(np.sqrt(mean_squared_error(y_valid, pred)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="catboost_rmse",
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_iterations = int(study.best_trial.user_attrs["best_iteration"])

    # Train the tuned candidate on train + validation. No early stopping is
    # retained in the serialized model, so the selector can refit it on all data.
    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    X_train_valid = train_valid_df[FEATURES]
    y_train_valid = train_valid_df[TARGET]

    final_params = {
        **best_params,
        "iterations": best_iterations,
        "loss_function": "RMSE",
        "eval_metric": "RMSE",
        "random_seed": 42,
        "verbose": 0,
        "allow_writing_files": False,
        "thread_count": -1,
    }
    model = CatBoostRegressor(**final_params)
    model.fit(X_train_valid, y_train_valid, verbose=False)

    pred_log = model.predict(X_test)
    rmse, mae_log, r2_log, mae_real = evaluate(y_test, pred_log)

    metrics_out.log_metric("rmse", rmse)
    metrics_out.log_metric("mae_log", mae_log)
    metrics_out.log_metric("r2_log", r2_log)
    metrics_out.log_metric("mae_real_ton_km2", mae_real)
    metrics_out.log_metric("optuna_best_validation_rmse", float(study.best_value))
    metrics_out.log_metric("optuna_trials", float(len(study.trials)))
    metrics_out.metadata["model_name"] = "CatBoost"
    metrics_out.metadata["best_params_json"] = json.dumps(final_params, default=str)

    model_artifact.metadata["framework"] = "CatBoost"
    model_artifact.metadata["best_params_json"] = json.dumps(final_params, default=str)
    joblib.dump(model, model_artifact.path)


@dsl.component(base_image=IMAGE_URI)
def train_lightgbm(
    dataset_in: Input[Dataset],
    model_artifact: Output[Model],
    metrics_out: Output[Metrics],
    n_trials: int = 30,
):
    import json
    import joblib
    import numpy as np
    import optuna
    import pandas as pd
    import lightgbm as lgb
    from lightgbm import LGBMRegressor
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from src.utils.features import apply_feature_engineering, FEATURES, TARGET

    def temporal_split(data: pd.DataFrame):
        data = data.copy()
        data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")
        data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
        data = (
            data.dropna(subset=["fecha", TARGET])
            .sort_values("fecha")
            .reset_index(drop=True)
        )

        unique_dates = np.sort(data["fecha"].unique())
        if len(unique_dates) < 3:
            raise ValueError("Se necesitan al menos 3 fechas distintas para train/validación/test.")

        train_pos = max(0, min(len(unique_dates) - 3, int(len(unique_dates) * 0.70) - 1))
        val_pos = max(train_pos + 1, min(len(unique_dates) - 2, int(len(unique_dates) * 0.85) - 1))

        train_end = unique_dates[train_pos]
        val_end = unique_dates[val_pos]

        train = data[data["fecha"] <= train_end]
        valid = data[(data["fecha"] > train_end) & (data["fecha"] <= val_end)]
        test = data[data["fecha"] > val_end]

        if train.empty or valid.empty or test.empty:
            raise ValueError(
                f"Split temporal inválido: train={len(train)}, valid={len(valid)}, test={len(test)}"
            )
        return train, valid, test

    def evaluate(y_true, pred_log):
        rmse = float(np.sqrt(mean_squared_error(y_true, pred_log)))
        mae_log = float(mean_absolute_error(y_true, pred_log))
        r2_log = float(r2_score(y_true, pred_log))
        mae_real = float(mean_absolute_error(np.expm1(y_true), np.expm1(pred_log)))
        return rmse, mae_log, r2_log, mae_real

    df = pd.read_parquet(dataset_in.path)
    df = apply_feature_engineering(df)
    train_df, valid_df, test_df = temporal_split(df)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_valid, y_valid = valid_df[FEATURES], valid_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    def objective(trial: optuna.Trial) -> float:
        max_depth = trial.suggest_int("max_depth", 4, 12)

        params = {
            "objective": "regression",
            "n_estimators": 4000,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "max_depth": max_depth,
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 1.0),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }

        candidate = LGBMRegressor(**params)
        candidate.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            eval_metric="rmse",
            callbacks=[
                lgb.early_stopping(stopping_rounds=150, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )
        pred = candidate.predict(X_valid, num_iteration=candidate.best_iteration_)
        best_iteration = candidate.best_iteration_ or params["n_estimators"]
        trial.set_user_attr("best_iteration", int(best_iteration))
        return float(np.sqrt(mean_squared_error(y_valid, pred)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="lightgbm_rmse",
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_iterations = int(study.best_trial.user_attrs["best_iteration"])

    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    X_train_valid = train_valid_df[FEATURES]
    y_train_valid = train_valid_df[TARGET]

    final_params = {
        **best_params,
        "objective": "regression",
        "n_estimators": best_iterations,
        "subsample_freq": 1,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model = LGBMRegressor(**final_params)
    model.fit(X_train_valid, y_train_valid)

    pred_log = model.predict(X_test)
    rmse, mae_log, r2_log, mae_real = evaluate(y_test, pred_log)

    metrics_out.log_metric("rmse", rmse)
    metrics_out.log_metric("mae_log", mae_log)
    metrics_out.log_metric("r2_log", r2_log)
    metrics_out.log_metric("mae_real_ton_km2", mae_real)
    metrics_out.log_metric("optuna_best_validation_rmse", float(study.best_value))
    metrics_out.log_metric("optuna_trials", float(len(study.trials)))
    metrics_out.metadata["model_name"] = "LightGBM"
    metrics_out.metadata["best_params_json"] = json.dumps(final_params, default=str)

    model_artifact.metadata["framework"] = "LightGBM"
    model_artifact.metadata["best_params_json"] = json.dumps(final_params, default=str)
    joblib.dump(model, model_artifact.path)


@dsl.component(base_image=IMAGE_URI)
def train_xgboost(
    dataset_in: Input[Dataset],
    model_artifact: Output[Model],
    metrics_out: Output[Metrics],
    n_trials: int = 30,
):
    import json
    import joblib
    import numpy as np
    import optuna
    import pandas as pd
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from xgboost import XGBRegressor
    from src.utils.features import apply_feature_engineering, FEATURES, TARGET

    def temporal_split(data: pd.DataFrame):
        data = data.copy()
        data["fecha"] = pd.to_datetime(data["fecha"], errors="coerce")
        data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
        data = (
            data.dropna(subset=["fecha", TARGET])
            .sort_values("fecha")
            .reset_index(drop=True)
        )

        unique_dates = np.sort(data["fecha"].unique())
        if len(unique_dates) < 3:
            raise ValueError("Se necesitan al menos 3 fechas distintas para train/validación/test.")

        train_pos = max(0, min(len(unique_dates) - 3, int(len(unique_dates) * 0.70) - 1))
        val_pos = max(train_pos + 1, min(len(unique_dates) - 2, int(len(unique_dates) * 0.85) - 1))

        train_end = unique_dates[train_pos]
        val_end = unique_dates[val_pos]

        train = data[data["fecha"] <= train_end]
        valid = data[(data["fecha"] > train_end) & (data["fecha"] <= val_end)]
        test = data[data["fecha"] > val_end]

        if train.empty or valid.empty or test.empty:
            raise ValueError(
                f"Split temporal inválido: train={len(train)}, valid={len(valid)}, test={len(test)}"
            )
        return train, valid, test

    def evaluate(y_true, pred_log):
        rmse = float(np.sqrt(mean_squared_error(y_true, pred_log)))
        mae_log = float(mean_absolute_error(y_true, pred_log))
        r2_log = float(r2_score(y_true, pred_log))
        mae_real = float(mean_absolute_error(np.expm1(y_true), np.expm1(pred_log)))
        return rmse, mae_log, r2_log, mae_real

    df = pd.read_parquet(dataset_in.path)
    df = apply_feature_engineering(df)
    train_df, valid_df, test_df = temporal_split(df)

    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_valid, y_valid = valid_df[FEATURES], valid_df[TARGET]
    X_test, y_test = test_df[FEATURES], test_df[TARGET]

    def objective(trial: optuna.Trial) -> float:
        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "tree_method": "hist",
            "n_estimators": 4000,
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "min_child_weight": trial.suggest_float("min_child_weight", 1e-2, 30.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.60, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.60, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 50.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
            "early_stopping_rounds": 150,
        }

        candidate = XGBRegressor(**params)
        candidate.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )
        pred = candidate.predict(X_valid)
        best_iteration = getattr(candidate, "best_iteration", None)
        trial.set_user_attr(
            "best_iteration",
            int(best_iteration + 1 if best_iteration is not None else params["n_estimators"]),
        )
        return float(np.sqrt(mean_squared_error(y_valid, pred)))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
        study_name="xgboost_rmse",
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_iterations = int(study.best_trial.user_attrs["best_iteration"])

    train_valid_df = pd.concat([train_df, valid_df], ignore_index=True)
    X_train_valid = train_valid_df[FEATURES]
    y_train_valid = train_valid_df[TARGET]

    final_params = {
        **best_params,
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "n_estimators": best_iterations,
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    model = XGBRegressor(**final_params)
    model.fit(X_train_valid, y_train_valid, verbose=False)

    pred_log = model.predict(X_test)
    rmse, mae_log, r2_log, mae_real = evaluate(y_test, pred_log)

    metrics_out.log_metric("rmse", rmse)
    metrics_out.log_metric("mae_log", mae_log)
    metrics_out.log_metric("r2_log", r2_log)
    metrics_out.log_metric("mae_real_ton_km2", mae_real)
    metrics_out.log_metric("optuna_best_validation_rmse", float(study.best_value))
    metrics_out.log_metric("optuna_trials", float(len(study.trials)))
    metrics_out.metadata["model_name"] = "XGBoost"
    metrics_out.metadata["best_params_json"] = json.dumps(final_params, default=str)

    model_artifact.metadata["framework"] = "XGBoost"
    model_artifact.metadata["best_params_json"] = json.dumps(final_params, default=str)
    joblib.dump(model, model_artifact.path)


@dsl.component(base_image=IMAGE_URI)
def select_and_register_best_model(
    project_id: str,
    region: str,
    dataset_in: Input[Dataset],
    catboost_model: Input[Model],
    catboost_metrics: Input[Metrics],
    lightgbm_model: Input[Model],
    lightgbm_metrics: Input[Metrics],
    xgboost_model: Input[Model],
    xgboost_metrics: Input[Metrics],
    model_display_name: str,
    final_model: Output[Model],
):
    import json
    import os
    import joblib
    import pandas as pd
    from google.cloud import aiplatform
    from src.utils.features import apply_feature_engineering, FEATURES, TARGET

    aiplatform.init(project=project_id, location=region)

    results = [
        {
            "name": "CatBoost",
            "rmse": float(catboost_metrics.metadata["rmse"]),
            "model_path": catboost_model.path,
            "params": catboost_metrics.metadata.get("best_params_json", "{}"),
        },
        {
            "name": "LightGBM",
            "rmse": float(lightgbm_metrics.metadata["rmse"]),
            "model_path": lightgbm_model.path,
            "params": lightgbm_metrics.metadata.get("best_params_json", "{}"),
        },
        {
            "name": "XGBoost",
            "rmse": float(xgboost_metrics.metadata["rmse"]),
            "model_path": xgboost_model.path,
            "params": xgboost_metrics.metadata.get("best_params_json", "{}"),
        },
    ]
    results.sort(key=lambda item: item["rmse"])
    best = results[0]
    print(f"Best model is {best['name']} with untouched-test RMSE: {best['rmse']}")

    best_model_obj = joblib.load(best["model_path"])

    # Refit the already-tuned model on all available data. The training
    # components serialize models without early-stopping dependencies.
    df = pd.read_parquet(dataset_in.path)
    df = apply_feature_engineering(df)
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = (
        df.dropna(subset=["fecha", TARGET])
        .sort_values("fecha")
        .reset_index(drop=True)
    )

    X_all = df[FEATURES]
    y_all = df[TARGET]
    best_model_obj.fit(X_all, y_all)

    # Keep the exact artifact contract consumed by infer_pipeline:
    #   model.joblib
    #   stats.parquet
    artifact_dir = final_model.path
    os.makedirs(artifact_dir, exist_ok=True)

    joblib.dump(best_model_obj, os.path.join(artifact_dir, "model.joblib"))

    # Keep mean/std for the current drift component and add extra distribution
    # descriptors without breaking infer_pipeline.
    stats = X_all.agg(["mean", "std", "min", "max", "median"]).T
    stats["q01"] = X_all.quantile(0.01)
    stats["q05"] = X_all.quantile(0.05)
    stats["q25"] = X_all.quantile(0.25)
    stats["q75"] = X_all.quantile(0.75)
    stats["q95"] = X_all.quantile(0.95)
    stats["q99"] = X_all.quantile(0.99)
    stats["missing_rate"] = X_all.isna().mean()
    stats.to_parquet(os.path.join(artifact_dir, "stats.parquet"))

    training_metadata = {
        "winner": best["name"],
        "test_rmse": best["rmse"],
        "best_params": json.loads(best["params"]),
        "candidates": [
            {"name": item["name"], "test_rmse": item["rmse"]}
            for item in results
        ],
        "features": list(FEATURES),
        "target": TARGET,
    }
    with open(os.path.join(artifact_dir, "training_metadata.json"), "w", encoding="utf-8") as file:
        json.dump(training_metadata, file, ensure_ascii=False, indent=2, default=str)

    final_model.metadata["winner"] = best["name"]
    final_model.metadata["test_rmse"] = best["rmse"]

    # final_model.uri is the GCS directory corresponding to final_model.path.
    # The infer pipeline will obtain this URI from the latest Model Registry entry.
    aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=final_model.uri,
        serving_container_image_uri=(
            "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest"
        ),
        sync=True,
    )


@dsl.pipeline(name="anchoveta-train-pipeline")
def train_pipeline(
    project_id: str,
    region: str,
    bq_table: str,
    model_display_name: str,
    experiment_name: str,
    n_trials: int = 3,
):
    extract_op = extract_data_from_bq(
        project_id=project_id,
        bq_table=bq_table,
    )

    # The three components remain independent and can run in parallel.
    catboost_op = train_catboost(
        dataset_in=extract_op.output,
        n_trials=n_trials,
    )

    lightgbm_op = train_lightgbm(
        dataset_in=extract_op.output,
        n_trials=n_trials,
    )

    xgboost_op = train_xgboost(
        dataset_in=extract_op.output,
        n_trials=n_trials,
    )

    select_and_register_best_model(
        project_id=project_id,
        region=region,
        dataset_in=extract_op.output,
        catboost_model=catboost_op.outputs["model_artifact"],
        catboost_metrics=catboost_op.outputs["metrics_out"],
        lightgbm_model=lightgbm_op.outputs["model_artifact"],
        lightgbm_metrics=lightgbm_op.outputs["metrics_out"],
        xgboost_model=xgboost_op.outputs["model_artifact"],
        xgboost_metrics=xgboost_op.outputs["metrics_out"],
        model_display_name=model_display_name,
    )


if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=train_pipeline,
        package_path="train_pipeline.json",
    )
