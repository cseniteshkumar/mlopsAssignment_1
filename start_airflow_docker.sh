#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
COMPOSE_FILE="$REPO_ROOT/docker-compose.airflow.yml"

echo "Starting Airflow with docker compose..."
# Bring up the container in detached mode
docker compose -f "$COMPOSE_FILE" up -d --remove-orphans

echo "Tailing Airflow logs (press Ctrl-C to stop)..."
docker compose -f "$COMPOSE_FILE" logs -f airflow
