#!/bin/bash

# https://www.docker.com/blog/how-to-use-the-postgres-docker-official-image/

export USER=${USER}
export TZ_LOCAL=America/Santiago

COMPOSE_PATH=/home/${USER}/repos_git/airflow/assembly/database/compose.yaml
PROJECT_NAME=${USER}_airflow_postgres

# docker compose -f ${COMPOSE_PATH} -p ${PROJECT_NAME} up --abort-on-container-exit --remove-orphans --force-recreate
docker compose -f ${COMPOSE_PATH} -p ${PROJECT_NAME} up --abort-on-container-exit --remove-orphans

# Execute using 
#   . ./main_docker_up.sh

# docker ps --filter name=toopazo* --filter status=running -aq | xargs docker stop
