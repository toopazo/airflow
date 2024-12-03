"""
Script para defiir los servicios basicos de la db
"""

# db_services.py
import json
from psycopg2.extras import execute_values
from datetime import datetime


def create_table_video(connection):
    with connection.cursor() as cursor:
        query = """
        CREATE TABLE video(
            id SERIAL PRIMARY KEY,
            name VARCHAR (255) UNIQUE NOT NULL,
            path VARCHAR (255) UNIQUE NOT NULL,
            timestamp TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        """
        cursor.execute(query)
        connection.commit()


def create_table_frame(connection):
    with connection.cursor() as cursor:
        query = """
        CREATE TABLE frame(
            id SERIAL PRIMARY KEY,
            video_id INT NOT NULL,
            count INT NOT NULL,
            path VARCHAR (255) UNIQUE NOT NULL,
            FOREIGN KEY (video_id) REFERENCES video(id)
        );
        """
        cursor.execute(query)
        connection.commit()


def create_table_inference(connection):
    with connection.cursor() as cursor:
        query = """
        CREATE TABLE inference(
            id SERIAL PRIMARY KEY,
            frame_id INT NOT NULL,
            bbox TEXT,
            kps TEXT,
            det_score FLOAT,
            landmark_3d_68 TEXT,
            pose TEXT,
            landmark_2d_106 TEXT,
            gender INT,
            age INT,
            embedding TEXT,
            FOREIGN KEY (frame_id) REFERENCES frame(id)
        );
        """
        cursor.execute(query)
        connection.commit()


def get_video(connection, origin, view_id):
    with connection.cursor() as cursor:
        # Modifica la consulta para obtener solo los registros del día actual
        cursor.execute(
            """
            SELECT
                aa.id,
                aa.view_video_urls
            FROM app_actions_view aa
            WHERE aa.origin = %s AND aa.view_id = %s
            """,
            (origin, view_id),
        )
        return cursor.fetchall()


def fetch_all_data(connection):
    with connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT 
                aa.id,
                aa."uuid",
                aa.action_type as view_type,
                aa."level" as view_level,
                aa.office_id as office,
                aa."zone",
                aa.paydesk,
                aa.cam_id as camera_number,
                aa.losses as recovery_amount,
                aa."discard" as reason_discard,
                aa."recovery" as is_non_recoverable,
                aa.processed as is_processed,
                count(ar.id) as total_scanned_products,
                jsonb_array_length(aa.product_crop_urls) AS total_disarmed_products,
                aa.reason_review_id as reason_review,
                aa.reason_review_comment,
                aa.ssn as reviewr_ssn,
                aa."user" as reviewr_user_name,
                aa.status as view_status,
                aa.image as view_image_url,
                aa.video_urls as view_video_urls,
                aa.product_crop_urls as disarmed_images_urls,
                aa.datetime as view_dt,
                aa.expiration_datetime as view_expiration_dt,
                aa.reviewd_dt,
                aa.partial_reviewd_dt
            FROM app_actions_view aa 
            LEFT JOIN app_actions_receipt ar ON ar.view_id = aa.id 
            GROUP BY aa.id
            """
        )
        return cursor.fetchall()


def insert_data_video(connection, row_list: list):
    with connection.cursor() as cursor:
        data_to_insert = []

        for row in row_list:
            data_to_insert.append(
                (
                    row[0],  # name VARCHAR (255) UNIQUE NOT NULL,
                    row[1],  # path VARCHAR (255) UNIQUE NOT NULL,
                )
            )

        query = """
            INSERT INTO video (
                name, path
            ) VALUES %s
        """

        execute_values(cursor, query, data_to_insert)
        connection.commit()


def find_id_video(connection, row_list: list):
    with connection.cursor() as cursor:
        id_list = []
        for row in row_list:
            query = (
                f"SELECT id FROM video WHERE name = '{row[0]}' AND path = '{row[1]}';"
            )
            cursor.execute(query)
            res = cursor.fetchall()
            # print(res)
            # [(1,)]
            if len(res) > 0:
                res = int(res[0][0])
                id_list.append(res)
    return id_list


def insert_data_frame(connection, row_list: list):
    with connection.cursor() as cursor:
        data_to_insert = []

        for row in row_list:
            data_to_insert.append(
                (
                    row[0],  # video_id,
                    row[1],  # count,
                    row[2],  # path,
                )
            )

        query = """
            INSERT INTO frame (
                video_id, count, path
            ) VALUES %s
        """

        execute_values(cursor, query, data_to_insert)
        connection.commit()


def find_id_frame(connection, row_list: list):
    with connection.cursor() as cursor:
        id_list = []
        for row in row_list:
            query = """
                SELECT id FROM frame
                WHERE
                    video_id = %s AND
                    count = %s AND
                    path = %s;
            """
            cursor.execute(query, row)
            res = cursor.fetchall()
            # print(res)
            # [(1,)]
            if len(res) > 0:
                res = int(res[0][0])
                id_list.append(res)
    return id_list


# def select_ident_current(connection, table: str):
#     with connection.cursor() as cursor:
#         query = f"SELECT IDENT_CURRENT('{table}') AS result FROM {table};"
#         cursor.execute(query)
#         row = cursor.fetchall()
#         # print(f"row {row}")
#         id = row[0]
#         return id


def insert_data_inference(connection, row_list: list):
    with connection.cursor() as cursor:
        data_to_insert = []

        for row in row_list:
            data_to_insert.append(
                (
                    row[0],  # frame_id
                    row[1],  # bbox,
                    row[2],  # kps,
                    row[3],  # det_score,
                    row[4],  # landmark_3d_68,
                    row[5],  # pose,
                    row[6],  # landmark_2d_106,
                    row[7],  # gender,
                    row[8],  # age,
                    row[9],  # embedding
                )
            )

        query = """
            INSERT INTO inference (
                frame_id, bbox, kps, det_score, landmark_3d_68, pose, landmark_2d_106, gender, age, embedding
            ) VALUES %s
        """

        execute_values(cursor, query, data_to_insert)
        connection.commit()


def find_id_inference(connection, row_list: list):
    with connection.cursor() as cursor:
        id_list = []
        for row in row_list:
            query = """
                SELECT id FROM inference
                WHERE
                    frame_id = %s AND
                    bbox = %s AND
                    kps = %s AND
                    det_score = %s AND
                    landmark_3d_68 = %s AND
                    pose = %s AND
                    landmark_2d_106 = %s AND
                    gender = %s AND
                    age = %s AND
                    embedding = %s;
            """
            cursor.execute(query, row)
            res = cursor.fetchall()
            # print(res)
            # [(1,)]
            if len(res) > 0:
                res = int(res[0][0])
                id_list.append(res)
    return id_list
