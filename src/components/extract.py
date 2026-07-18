import kfp.dsl as dsl

@dsl.component(
    packages_to_install=["google-cloud-bigquery", "pandas", "db-dtypes", "pyarrow"]
)
def extract_data_from_bq(
    project_id: str,
    bq_table: str,
    dataset_out: dsl.OutputPath("Dataset")
):
    import pandas as pd
    from google.cloud import bigquery
    
    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{bq_table}`"
    print(f"Extracting data from: {bq_table}")
    
    df = client.query(query).to_dataframe()
    df.to_parquet(dataset_out, index=False)
    print(f"Extracted {len(df)} rows.")
