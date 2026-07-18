import os
from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model, Metrics

# Hardcode temporarily to bypass local env issues for KFP compilation
IMAGE_URI = "us-central1-docker.pkg.dev/anchoveta/mlops-repo/anchoveta-pipeline:latest"
print(f"DEBUG: Compiling with IMAGE_URI={IMAGE_URI}")

@dsl.component(base_image=IMAGE_URI)
def extract_data_from_bq(
    project_id: str,
    bq_table: str,
    dataset_out: Output[Dataset]
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
    # Convert dbdate and dbtime to standard strings before saving to parquet
    # BigQuery Storage API returns dbdate which Pandas/PyArrow cannot serialize natively.
    for col in df.select_dtypes(include=['dbdate', 'object']).columns:
        df[col] = df[col].astype(str)
        
    df.to_parquet(dataset_out.path, index=False)

@dsl.component(base_image=IMAGE_URI)
def train_catboost(
    dataset_in: Input[Dataset],
    model_artifact: Output[Model],
    metrics_out: Output[Metrics]
):
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score
    from catboost import CatBoostRegressor
    from src.utils.features import apply_feature_engineering, FEATURES, TARGET
    
    df = pd.read_parquet(dataset_in.path)
    df = apply_feature_engineering(df)
    
    corte = df["fecha"].quantile(0.8)
    train_df = df[df["fecha"] <= corte]
    test_df  = df[df["fecha"] > corte]
    
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]
    
    model = CatBoostRegressor(iterations=3000, depth=6, learning_rate=0.03, l2_leaf_reg=5, loss_function="RMSE", random_seed=42, verbose=0, early_stopping_rounds=200)
    model.fit(X_train, y_train, eval_set=(X_test, y_test))
    
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    
    metrics_out.log_metric("rmse", float(rmse))
    metrics_out.log_metric("model_name", "CatBoost")
    joblib.dump(model, model_artifact.path)

@dsl.component(base_image=IMAGE_URI)
def train_lightgbm(
    dataset_in: Input[Dataset],
    model_artifact: Output[Model],
    metrics_out: Output[Metrics]
):
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score
    from lightgbm import LGBMRegressor
    from src.utils.features import apply_feature_engineering, FEATURES, TARGET
    
    df = pd.read_parquet(dataset_in.path)
    df = apply_feature_engineering(df)
    
    corte = df["fecha"].quantile(0.8)
    train_df = df[df["fecha"] <= corte]
    test_df  = df[df["fecha"] > corte]
    
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]
    
    model = LGBMRegressor(n_estimators=3000, max_depth=6, learning_rate=0.03, num_leaves=31, reg_lambda=5, random_state=42, verbose=-1)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)])
    
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    
    metrics_out.log_metric("rmse", float(rmse))
    metrics_out.log_metric("model_name", "LightGBM")
    joblib.dump(model, model_artifact.path)

@dsl.component(base_image=IMAGE_URI)
def train_xgboost(
    dataset_in: Input[Dataset],
    model_artifact: Output[Model],
    metrics_out: Output[Metrics]
):
    import pandas as pd
    import numpy as np
    import joblib
    from sklearn.metrics import mean_squared_error, r2_score
    from xgboost import XGBRegressor
    from src.utils.features import apply_feature_engineering, FEATURES, TARGET
    
    df = pd.read_parquet(dataset_in.path)
    df = apply_feature_engineering(df)
    
    corte = df["fecha"].quantile(0.8)
    train_df = df[df["fecha"] <= corte]
    test_df  = df[df["fecha"] > corte]
    
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]
    
    model = XGBRegressor(n_estimators=3000, max_depth=6, learning_rate=0.03, reg_lambda=5, random_state=42, verbosity=0, early_stopping_rounds=200)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    
    pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, pred))
    
    metrics_out.log_metric("rmse", float(rmse))
    metrics_out.log_metric("model_name", "XGBoost")
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
    final_model: Output[Model]
):
    import pandas as pd
    import joblib
    import os
    from google.cloud import aiplatform
    from src.utils.features import apply_feature_engineering, FEATURES, TARGET
    
    aiplatform.init(project=project_id, location=region)
    
    # Determine best model based on RMSE
    results = [
        {"name": "CatBoost", "rmse": catboost_metrics.metadata["rmse"], "model_path": catboost_model.path},
        {"name": "LightGBM", "rmse": lightgbm_metrics.metadata["rmse"], "model_path": lightgbm_model.path},
        {"name": "XGBoost", "rmse": xgboost_metrics.metadata["rmse"], "model_path": xgboost_model.path}
    ]
    
    results.sort(key=lambda x: x["rmse"])
    best = results[0]
    print(f"Best model is {best['name']} with RMSE: {best['rmse']}")
    
    # Load best model
    best_model_obj = joblib.load(best["model_path"])
    
    # Refit on all data
    df = pd.read_parquet(dataset_in.path)
    df = apply_feature_engineering(df)
    X_all, y_all = df[FEATURES], df[TARGET]
    best_model_obj.fit(X_all, y_all)
    
    # Save final model explicitly as model.joblib in the artifact directory
    local_dir = os.path.dirname(final_model.path)
    joblib.dump(best_model_obj, os.path.join(local_dir, "model.joblib"))
    
    # Save stats for drift
    stats = X_all.agg(['mean', 'std']).T
    stats.to_parquet(os.path.join(local_dir, "stats.parquet"))
    
    # Register to Vertex Model Registry
    aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=local_dir,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
        sync=True
    )

@dsl.pipeline(name="anchoveta-train-pipeline")
def train_pipeline(
    project_id: str,
    region: str,
    bq_table: str,
    model_display_name: str,
    experiment_name: str
):
    extract_op = extract_data_from_bq(project_id=project_id, bq_table=bq_table)
    
    catboost_op = train_catboost(dataset_in=extract_op.output)
    lightgbm_op = train_lightgbm(dataset_in=extract_op.output)
    xgboost_op = train_xgboost(dataset_in=extract_op.output)
    
    select_op = select_and_register_best_model(
        project_id=project_id,
        region=region,
        dataset_in=extract_op.output,
        catboost_model=catboost_op.outputs["model_artifact"],
        catboost_metrics=catboost_op.outputs["metrics_out"],
        lightgbm_model=lightgbm_op.outputs["model_artifact"],
        lightgbm_metrics=lightgbm_op.outputs["metrics_out"],
        xgboost_model=xgboost_op.outputs["model_artifact"],
        xgboost_metrics=xgboost_op.outputs["metrics_out"],
        model_display_name=model_display_name
    )

if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=train_pipeline, package_path="train_pipeline.json")
