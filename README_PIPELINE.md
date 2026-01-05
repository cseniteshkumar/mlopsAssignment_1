Airflow DAG and how to run the project's pipeline

This project includes an Airflow DAG at `dags/pipeline_dag.py` and helper
wrappers in `src/airflow_tasks.py`.

Overview
- The DAG is intentionally thin: it calls the project's CLI (`src/pipeline.py`) via a BashOperator. This avoids import/PYTHONPATH problems at DAG-parse time.
- Alternately, `src/airflow_tasks.py` contains Python-callable wrappers that persist intermediate artifacts to disk. These can be used with PythonOperator if you configure Airflow's PYTHONPATH to include the project root.

Quick run (recommended approach using the included DAG)
1. Set environment variables so Airflow can discover the repo and DAGs if needed:

```bash
export AIRFLOW_HOME=~/airflow
export PROJECT_ROOT="/media/niteshkumar/SSD_Store_0_nvme/allPythoncodesWithPipEnv/BitsLearning/MLOps_Assignment/Assignment_1"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
```

2. Install Airflow in a virtualenv or conda env (pick a compatible version):

```bash
# Example (pick a stable version for your environment):
pip install 'apache-airflow==2.6.3'
```

3. Initialize Airflow DB and start services:

```bash
airflow db init
airflow users create --username admin --firstname Admin --lastname User --role Admin --email admin@example.com
airflow webserver --port 8080
# in another terminal
airflow scheduler
```

4. Ensure the DAG file `dags/pipeline_dag.py` is visible to Airflow. If your
Airflow `dags/` folder is not the repository `dags/` folder, copy or symlink the
file there.

5. Trigger the DAG from the UI or CLI:

```bash
airflow dags list
airflow dags trigger mlops_assignment_pipeline
```

Notes and troubleshooting
- If the DAG shows import errors in the UI, check that Airflow's worker
  environment can import `src` (set `PYTHONPATH` or use BashOperator approach).
- The DAG included in this repo uses a BashOperator that runs the CLI command
  `python <repo>/src/pipeline.py`. This should work without modifying Airflow's
  Python path as long as Python in the worker can run the script at that path.
- If you prefer PythonOperator, update Airflow's PYTHONPATH to include the
  project root and switch the DAG to call functions in `src.airflow_tasks`.

Optional: full Airflow install by this agent
- Installing and running Airflow in this environment is possible but may be
  time-consuming and require system packages. Tell me if you want me to attempt
  that here and I'll proceed.

Docker Compose (recommended)
---------------------------------
If you don't want to install Airflow into your host Python, the easiest and
most reproducible local option is to run Airflow inside the official Docker
image. This repository contains `docker-compose.airflow.yml` and a small helper
script `start_airflow_docker.sh` that will start Airflow and mount this repo
so your DAGs are visible.

Quick steps:

```bash
# Make sure you have Docker and Docker Compose installed
chmod +x start_airflow_docker.sh
./start_airflow_docker.sh
```

The web UI will be available at http://localhost:8080 (default credentials in
the compose file are `admin` / `admin`). This Docker setup uses `airflow
standalone` inside the official image for a quick, single-container local
experience. For production or multi-node setups use the official Airflow
docker-compose example or Kubernetes deployment.

