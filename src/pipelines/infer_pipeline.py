import os
from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model

IMAGE_URI = os.getenv("DOCKER_IMAGE_URI", "python:3.10-slim")

@dsl.component(base_image=IMAGE_URI)
def extract_infer_data(
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
def predict_and_drift(
    project_id: str,
    region: str,
    dataset_in: Input[Dataset],
    model_display_name: str,
    bq_pred_table: str,
    bq_drift_table: str
):
    import pandas as pd
    import numpy as np
    import joblib
    import os
    from google.cloud import aiplatform
    from google.cloud import bigquery
    from scipy.stats import ks_2samp
    
    from src.utils.features import apply_feature_engineering, FEATURES, TARGET
    
    aiplatform.init(project=project_id, location=region)
    
    # 1. Fetch Latest Model
    models = aiplatform.Model.list(filter=f'display_name="{model_display_name}"', order_by="create_time desc")
    if not models:
        raise ValueError(f"No model found with name: {model_display_name}")
    latest_model = models[0]
    
    # Download artifact
    import tempfile
    import subprocess
    tmp_dir = tempfile.mkdtemp()
    subprocess.run(["gsutil", "cp", "-r", f"{latest_model.uri}/*", tmp_dir])
    
    # load first file found (the model)
    model_file = [f for f in os.listdir(tmp_dir) if f.endswith(".joblib") or f.endswith(".pkl") or "model" in f][0]
    model = joblib.load(os.path.join(tmp_dir, model_file))
    
    # 2. Extract Data & Feature Engineering
    df = pd.read_parquet(dataset_in.path)
    df = apply_feature_engineering(df)
    
    X_infer = df[FEATURES]
    
    # 3. Calculate Drift (simplified)
    # ponytail: naive KS stat vs saved means. Real PSI needs binning, KS is one line.
    stats = pd.read_parquet(os.path.join(tmp_dir, "stats.parquet"))
    drift_metrics = []
    
    for feat in FEATURES:
        mean_train = stats.loc[feat, 'mean']
        std_train = stats.loc[feat, 'std']
        
        # approximate a normal distribution from train stats to compare
        train_approx = np.random.normal(mean_train, std_train, size=1000)
        stat, pval = ks_2samp(train_approx, X_infer[feat].dropna())
        
        drift_metrics.append({
            "feature": feat,
            "ks_stat": stat,
            "p_value": pval,
            "drift_detected": bool(pval < 0.05)
        })
        
    drift_df = pd.DataFrame(drift_metrics)
    drift_df["inference_time"] = pd.Timestamp.now()
    
    # 4. Predict
    preds = model.predict(X_infer)
    df["predicted_log_densidad"] = preds
    df["inference_time"] = pd.Timestamp.now()
    
    # 5. Write to BQ
    bq_client = bigquery.Client(project=project_id)
    
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    # Write preds
    bq_client.load_table_from_dataframe(
        df[["cell_id", "fecha", "predicted_log_densidad", "inference_time"]],
        bq_pred_table,
        job_config=job_config
    ).result()
    
    # Write drift
    bq_client.load_table_from_dataframe(
        drift_df,
        bq_drift_table,
        job_config=job_config
    ).result()

@dsl.pipeline(name="anchoveta-infer-pipeline")
def infer_pipeline(
    project_id: str,
    region: str,
    bq_table: str,
    model_display_name: str,
    bq_pred_table: str,
    bq_drift_table: str
):
    extract_op = extract_infer_data(project_id=project_id, bq_table=bq_table)
    predict_op = predict_and_drift(
        project_id=project_id,
        region=region,
        dataset_in=extract_op.output,
        model_display_name=model_display_name,
        bq_pred_table=bq_pred_table,
        bq_drift_table=bq_drift_table
    )

if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=infer_pipeline, package_path="infer_pipeline.json")
