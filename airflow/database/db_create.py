import os
from pathlib import Path
from datetime import datetime
import pandas as pd
from tqdm import tqdm
from airflow.database.db_config import connect_to_database, db_airflow
from airflow.database.db_services import (
    create_table_video,
    create_table_frame,
    create_table_inference,
    create_table_sequence,
)


class DbOps:
    def __init__(self):
        pass

    def create_tables(self):
        conn = connect_to_database(db_airflow)
        if conn:
            try:
                create_table_video(conn)
            except Exception as e:
                print(f"Error interacting with {db_airflow['database']}: {e}")

            try:
                create_table_frame(conn)
            except Exception as e:
                print(f"Error interacting with {db_airflow['database']}: {e}")

            try:
                create_table_inference(conn)
            except Exception as e:
                print(f"Error interacting with {db_airflow['database']}: {e}")

            try:
                create_table_sequence(conn)
            except Exception as e:
                print(f"Error interacting with {db_airflow['database']}: {e}")

            conn.close()
        else:
            print(f"No connection to {db_airflow['database']}.")


if __name__ == "__main__":
    do = DbOps()
    do.create_tables()
