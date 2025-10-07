from __future__ import annotations

import os
from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator


def task_bootstrap_opensearch():
    from ai_finance.search.opensearch_client import ensure_index
    ensure_index()


def task_bootstrap_qdrant():
    from ai_finance.embedding.encoder import EmbeddingEncoder
    from ai_finance.storage.qdrant_client import ensure_collection
    enc = EmbeddingEncoder()
    ensure_collection(enc.dimension)


def task_index_pipeline(**context):
    from ai_finance.pipeline.index_pipeline import run_index_pipeline
    prefix = os.getenv("AWS_S3_PREFIX")
    run_index_pipeline(s3_prefix=prefix)


with DAG(
    dag_id="billing_system_information_retreival_and_match",
    schedule_interval=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args={"owner": "data-platform"},
) as dag:
    bootstrap_os = PythonOperator(task_id="bootstrap_opensearch", python_callable=task_bootstrap_opensearch)
    bootstrap_qd = PythonOperator(task_id="bootstrap_qdrant", python_callable=task_bootstrap_qdrant)
    index_run = PythonOperator(task_id="run_index_pipeline", python_callable=task_index_pipeline)

    [bootstrap_os, bootstrap_qd] >> index_run


