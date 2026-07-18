import os
from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model

IMAGE_URI = os.getenv("DOCKER_IMAGE_URI", "python:3.10-slim") # To be replaced at compile time or run time

@dsl.component(base_image=IMAGE_URI)
def extract_data_from_bq(
    project_id: str,
    bq_table: str,
    dataset_out: Output[Dataset]
):
    from google.cloud import bigquery
    import pandas as pd
    
    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{bq_table}`"
    df = client.query(query).to_dataframe()
    df.to_parquet(dataset_out.path, index=False)

@dsl.component(base_image=IMAGE_URI)
def train_and_register_model(
    project_id: str,
    region: str,
    dataset_in: Input[Dataset],
    model_display_name: str,
    experiment_name: str,
    model_artifact: Output[Model]
):
    import pandas as pd
    import numpy as np
    import time
    import joblib
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import TimeSeriesSplit
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor
    from google.cloud import aiplatform
    
    # Import shared features
    from src.utils.features import apply_feature_engineering, FEATURES, TARGET
    
    aiplatform.init(project=project_id, location=region, experiment=experiment_name)
    
    df = pd.read_parquet(dataset_in.path)
    df = apply_feature_engineering(df)
    
    corte = df["fecha"].quantile(0.8)
    train_df = df[df["fecha"] <= corte]
    test_df  = df[df["fecha"] > corte]
    
    X_train, y_train = train_df[FEATURES], train_df[TARGET]
    X_test,  y_test  = test_df[FEATURES],  test_df[TARGET]
    
    modelos = {
        "CatBoost": CatBoostRegressor(iterations=3000, depth=6, learning_rate=0.03, l2_leaf_reg=5, loss_function="RMSE", random_seed=42, verbose=0, early_stopping_rounds=200),
        "LightGBM": LGBMRegressor(n_estimators=3000, max_depth=6, learning_rate=0.03, num_leaves=31, reg_lambda=5, random_state=42, verbose=-1),
        "XGBoost": XGBRegressor(n_estimators=3000, max_depth=6, learning_rate=0.03, reg_lambda=5, random_state=42, verbosity=0, early_stopping_rounds=200)
    }
    
    resultados = []
    
    # ponytail: naive loop, add hyperparameter tuning if default models drift
    for nombre, model in modelos.items():
        aiplatform.start_run(run=f"{nombre}-{int(time.time())}")
        t0 = time.time()
        
        if nombre == "CatBoost":
            model.fit(X_train, y_train, eval_set=(X_test, y_test))
        elif nombre == "XGBoost":
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
        else:
            model.fit(X_train, y_train, eval_set=[(X_test, y_test)]) # lgbm supports eval_set
            
        pred = model.predict(X_test)
        elapsed = time.time() - t0
        rmse = np.sqrt(mean_squared_error(y_test, pred))
        r2 = r2_score(y_test, pred)
        
        aiplatform.log_metrics({"rmse": rmse, "r2": r2, "time_s": elapsed})
        aiplatform.end_run()
        
        resultados.append({"Modelo": nombre, "RMSE": rmse, "Model_Obj": model})
    
    resultados_df = pd.DataFrame(resultados).sort_values("RMSE")
    mejor_nombre = resultados_df.iloc[0]["Modelo"]
    best_model = resultados_df.iloc[0]["Model_Obj"]
    print(f"Best model: {mejor_nombre}")
    
    # Walk-forward validation on best model (simplified)
    # ponytail: re-training on all data directly for final model. 
    X_all, y_all = df[FEATURES], df[TARGET]
    best_model.fit(X_all, y_all)
    
    # Save model and stats in the same directory
    import os
    model_path = model_artifact.path
    joblib.dump(best_model, model_path)
    
    artifact_dir = os.path.dirname(model_artifact.uri)
    
    # Save training stats for drift in the same local directory before upload
    local_dir = os.path.dirname(model_artifact.path)
    stats = X_all.agg(['mean', 'std']).T
    stats.to_parquet(os.path.join(local_dir, "stats.parquet"))
    
    # Register model (uploads the whole directory)
    aiplatform.Model.upload(
        display_name=model_display_name,
        artifact_uri=artifact_dir,
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
    train_op = train_and_register_model(
        project_id=project_id,
        region=region,
        dataset_in=extract_op.output,
        model_display_name=model_display_name,
        experiment_name=experiment_name
    )

if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=train_pipeline, package_path="train_pipeline.json")
