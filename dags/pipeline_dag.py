"""Airflow DAG that orchestrates the project's ML pipeline.

This DAG uses PythonOperator to call thin wrappers in `src.airflow_tasks`.

Notes for running:
- Ensure Airflow's Python environment has the project's root on PYTHONPATH so
  `import src.airflow_tasks` works (for example, set `AIRFLOW__CORE__DAGS_FOLDER`
  or edit `sys.path` in the DAG if needed).
"""
from datetime import timedelta
from pathlib import Path

from airflow import DAG
from airflow.utils.dates import days_ago
from airflow.operators.bash import BashOperator

DEFAULT_ARGS = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 0,
}

with DAG(
    dag_id="mlops_assignment_pipeline",
    default_args=DEFAULT_ARGS,
    description="End-to-end pipeline DAG using project stage wrappers",
    schedule_interval=None,
    start_date=days_ago(1),
    catchup=False,
    default_view="graph",
) as dag:

    ARTIFACTS = str(Path(__file__).resolve().parents[1] / "artifacts")
    MODEL_OUT = str(Path(__file__).resolve().parents[1] / "models" / "model.joblib")

    # Use BashOperators that run the project's CLI script. This avoids issues
    # with Airflow's Python import path at parse-time and keeps the DAG thin.
    ROOT = Path(__file__).resolve().parents[1]
    PIPE_CMD = f"python {ROOT}/src/pipeline.py --dataset DataSet --output {ROOT}/models/model.joblib"

    t_run_pipeline = BashOperator(
        task_id="run_pipeline_cli",
        bash_command=PIPE_CMD,
    )

    t_run_pipeline
