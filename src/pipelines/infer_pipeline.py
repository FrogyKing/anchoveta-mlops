import os
from kfp import dsl
from kfp import compiler
from kfp.dsl import Input, Output, Dataset, Model

# Hardcode temporarily to bypass local env issues for KFP compilation
IMAGE_URI = "us-central1-docker.pkg.dev/anchoveta/mlops-repo/anchoveta-pipeline:latest"
print(f"DEBUG: Compiling with IMAGE_URI={IMAGE_URI}")

@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=["google-cloud-bigquery==3.42.2", "pandas<3.0.0", "db-dtypes", "pyarrow"]
)
def extract_infer_data(
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
    for col in df.select_dtypes(include=['dbdate', 'object']).columns:
        df[col] = df[col].astype(str)
        
    df.to_parquet(dataset_out.path, index=False)

@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=[
        "google-cloud-aiplatform==1.161.0",
        "pandas<3.0.0",
        "scikit-learn==1.9.0",
        "pyarrow",
        "xgboost==3.3.0",
        "lightgbm==4.6.0",
        "catboost==1.2.10"
    ]
)
def predict(
    project_id: str,
    region: str,
    dataset_in: Input[Dataset],
    model_display_name: str,
    predictions_out: Output[Dataset],
    model_artifact_out: Output[Model]
):
    import pandas as pd
    import joblib
    import os
    from google.cloud import aiplatform
    from src.utils.features import apply_feature_engineering, FEATURES
    
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
    
    # load model
    model_file = [f for f in os.listdir(tmp_dir) if f.endswith(".joblib") or f.endswith(".pkl") or f.endswith(".bst")][0]
    model = joblib.load(os.path.join(tmp_dir, model_file))
    
    # Optional: Save downloaded model into Output[Model] artifact if downstream steps need it
    joblib.dump(model, model_artifact_out.path)
    # Also propagate the stats file to the artifact dir for the drift component
    subprocess.run(["cp", os.path.join(tmp_dir, "stats.parquet"), os.path.join(os.path.dirname(model_artifact_out.path), "stats.parquet")])
    
    # 2. Extract Data & Feature Engineering
    df = pd.read_parquet(dataset_in.path)
    df = apply_feature_engineering(df)
    
    X_infer = df[FEATURES]
    
    # 4. Predict
    preds = model.predict(X_infer)
    df["predicted_log_densidad"] = preds
    df["inference_time"] = pd.Timestamp.now()
    
    # Save predictions
    df[["cell_id", "fecha", "predicted_log_densidad", "inference_time"]].to_parquet(predictions_out.path, index=False)

@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=[
        "google-cloud-bigquery==3.42.2",
        "pandas<3.0.0",
        "pyarrow"
    ]
)
def save_predictions_to_bq(
    project_id: str,
    predictions_in: Input[Dataset],
    bq_pred_table: str
):
    import pandas as pd
    from google.cloud import bigquery
    
    df = pd.read_parquet(predictions_in.path)
    
    bq_client = bigquery.Client(project=project_id)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    
    bq_client.load_table_from_dataframe(
        df,
        bq_pred_table,
        job_config=job_config
    ).result()

@dsl.component(
    base_image=IMAGE_URI,
    packages_to_install=[
        "google-cloud-bigquery==3.42.2",
        "pandas<3.0.0",
        "scipy==1.18.0",
        "pyarrow"
    ]
)
def calculate_and_save_drift(
    project_id: str,
    dataset_in: Input[Dataset],
    model_artifact_in: Input[Model],
    bq_drift_table: str
):
    import pandas as pd
    import numpy as np
    import os
    from scipy.stats import ks_2samp
    from google.cloud import bigquery
    from src.utils.features import apply_feature_engineering, FEATURES
    
    # 1. Prepare Inference Data
    df = pd.read_parquet(dataset_in.path)
    df = apply_feature_engineering(df)
    X_infer = df[FEATURES]
    
    # 2. Load stats from model artifact directory
    stats_path = os.path.join(os.path.dirname(model_artifact_in.path), "stats.parquet")
    stats = pd.read_parquet(stats_path)
    
    drift_metrics = []
    
    # 3. Calculate KS Drift
    # ponytail: naive KS stat vs saved means. Real PSI needs binning, KS is one line.
    for feat in FEATURES:
        mean_train = stats.loc[feat, 'mean']
        std_train = stats.loc[feat, 'std']
        
        train_approx = np.random.normal(mean_train, std_train, size=1000)
        stat, pval = ks_2samp(train_approx, X_infer[feat].dropna())
        
        drift_metrics.append({
            "feature": feat,
            "ks_stat": float(stat),
            "p_value": float(pval),
            "drift_detected": bool(pval < 0.05)
        })
        
    drift_df = pd.DataFrame(drift_metrics)
    drift_df["inference_time"] = pd.Timestamp.now()
    
    # 4. Save to BQ
    bq_client = bigquery.Client(project=project_id)
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_APPEND")
    
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
    
    predict_op = predict(
        project_id=project_id,
        region=region,
        dataset_in=extract_op.output,
        model_display_name=model_display_name
    )
    
    save_preds_op = save_predictions_to_bq(
        project_id=project_id,
        predictions_in=predict_op.outputs["predictions_out"],
        bq_pred_table=bq_pred_table
    )
    
    drift_op = calculate_and_save_drift(
        project_id=project_id,
        dataset_in=extract_op.output,
        model_artifact_in=predict_op.outputs["model_artifact_out"],
        bq_drift_table=bq_drift_table
    )

if __name__ == "__main__":
    compiler.Compiler().compile(pipeline_func=infer_pipeline, package_path="infer_pipeline.json")
