#!/bin/bash

export TZ_LOCAL=America/Santiago
export CODE_PATH=airflow/database/compose.yaml
export PROJECT_NAME=${USER}_airflow

source .env

docker compose -f ${CODE_PATH} -p ${PROJECT_NAME} up  --abort-on-container-exit --remove-orphans

# Execute using 
#   . ./compose.sh