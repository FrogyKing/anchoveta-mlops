import os
from dotenv import load_dotenv
from google.cloud import aiplatform

load_dotenv()

PROJECT_ID = os.getenv("GCP_PROJECT")
REGION = os.getenv("REGION")
PIPELINE_ROOT = os.getenv("GCS_PIPELINE_ROOT")
BQ_TRAIN_TABLE = os.getenv("BQ_TRAIN_TABLE")
MODEL_DISPLAY_NAME = os.getenv("MODEL_DISPLAY_NAME")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME")

aiplatform.init(project=PROJECT_ID, location=REGION)

job = aiplatform.PipelineJob(
    display_name="anchoveta-train",
    template_path="src/pipelines/train_pipeline.json",
    pipeline_root=PIPELINE_ROOT,
    parameter_values={
        "project_id": PROJECT_ID,
        "region": REGION,
        "bq_table": BQ_TRAIN_TABLE,
        "model_display_name": MODEL_DISPLAY_NAME,
        "experiment_name": EXPERIMENT_NAME,
    },
    enable_caching=False
)

job.submit()
print(f"Submitted training pipeline: {job.resource_name}")
