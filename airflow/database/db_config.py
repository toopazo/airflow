"""
Script para definir parametros basicos de la db
"""

import os
import logging
import psycopg2
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env.

# Configuración del logger
logging.basicConfig(
    level=logging.ERROR,  # Cambia a DEBUG para más detalles si lo deseas
    format="%(asctime)s - %(levelname)s - %(message)s",
)


POSTGRES_USER = os.environ["POSTGRES_USER"]
POSTGRES_PASSWORD = os.environ["POSTGRES_PASSWORD"]
POSTGRES_DB = os.environ["POSTGRES_DB"]

# Configuración de las bases de datos

exposed_port = 54322
db_airflow = {
    "host": "localhost",
    "port": exposed_port,
    "user": POSTGRES_USER,
    "password": POSTGRES_PASSWORD,
    "database": POSTGRES_DB,
}


def connect_to_database(config):
    connection_params = {
        key: config[key]
        for key in ["host", "port", "user", "password", "database"]
        if key in config
    }
    try:
        connection = psycopg2.connect(**connection_params)
        logging.info(
            "Successful connection %s:%s",
            connection_params["host"],
            connection_params["port"],
        )
        return connection
    except psycopg2.Error as err:
        logging.error("Error connecting: %s", err)
        return None


def connect_and_execute(service_fnct, row_list):
    print("connect_and_execute ..")
    conn = connect_to_database(db_airflow)
    res_list = []
    if conn:
        try:
            res_list = service_fnct(conn, row_list)
        except Exception as e:
            print(f"Error interacting with {db_airflow['database']}: {e}")
        conn.close()
    else:
        print(f"No connection to {db_airflow['database']}.")
    return res_list
