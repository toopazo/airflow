#!/bin/bash

export USER=${USER}
export TZ_LOCAL=America/Santiago
export CODE_PATH=/home/${USER}/repos_git/airflow/docker/compose.yaml
export PROJECT_NAME=${USER}_airflow


docker compose -f ${CODE_PATH} -p ${PROJECT_NAME} down

# Execute using 
#   . ./compose.sh