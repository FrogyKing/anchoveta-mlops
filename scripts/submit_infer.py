import os
from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT")
REGION = os.getenv("REGION")
PIPELINE_ROOT = os.getenv("GCS_PIPELINE_ROOT")
BQ_INFER_TABLE = os.getenv("BQ_INFER_TABLE")
BQ_PRED_TABLE = os.getenv("BQ_PRED_TABLE")
BQ_DRIFT_TABLE = os.getenv("BQ_DRIFT_TABLE")
MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME")

aiplatform.init(project=PROJECT_ID, location=REGION)

job = aiplatform.PipelineJob(
    display_name="anchoveta-infer",
    template_path="infer_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "project_id": PROJECT_ID,
        "region": REGION,
        "bq_table": BQ_INFER_TABLE,
        "model_display_name": MODEL_DISPLAY_NAME,
        "bq_pred_table": BQ_PRED_TABLE,
        "bq_drift_table": BQ_DRIFT_TABLE
    },
    enable_caching=False
)

job.submit()
print(f"Submitted inference pipeline: {job.resource_name}")
