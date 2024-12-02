import os
import re
from ast import literal_eval
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pandas as pd

from assembly.database.db_config import (
    connect_to_database,
    db_airflow,
    connect_and_execute,
)
from assembly.database.db_services import (
    insert_data_video,
    find_id_video,
    insert_data_frame,
    find_id_frame,
    insert_data_inference,
    find_id_inference,
)


class ProcessData:
    def __init__(self):
        pass

    def add_video_data_to_database(self, video_path: Path, data_dir: Path):
        assert data_dir.is_dir()
        assert video_path.is_file()

        df_path = data_dir / "dataframe.csv"
        assert df_path.is_file()
        imaged_dir = data_dir / "imaged"
        assert imaged_dir.is_dir()
        videod_path = data_dir / "imaged.mp4"
        assert videod_path.is_file()

        df = pd.read_csv(df_path, index_col=0)
        video_name = df["name"].values[0]

        # name VARCHAR (255) UNIQUE NOT NULL,
        # path VARCHAR (255) UNIQUE NOT NULL,

        self.insert_video(row_list=[[str(video_name), str(video_path)]])
        video_id = self.find_video(row_list=[[str(video_name), str(video_path)]])
        if len(video_id) == 0:
            raise RuntimeError
        video_id = video_id[0]
        print(f"video id {video_id}")

        # for cnt in set(df["cnt"].values):
        #     cnt = int(cnt)
        #     frame_path = Path(
        #         f"/home/{user}/repos_git/airflow/insightface/output/{name}/imaged/imaged_{str(cnt).zfill(4)}.png"
        #     )
        #     # print(frame_path)
        #     assert frame_path.is_file()
        #     # video_id INT NOT NULL,
        #     # count INT NOT NULL,
        #     # path VARCHAR (255) UNIQUE NOT NULL,
        #     frame_id = self.insert_frame(row_list=[[video_id, cnt, str(frame_path)]])

        # return

        for ix, row in tqdm(df.iterrows(), total=df.shape[0]):
            _ = ix

            (
                counter,
                bbox,
                kps,
                det_score,
                landmark_3d_68,
                pose,
                landmark_2d_106,
                gender,
                age,
                embedding,
            ) = self.parse_video_row(row)

            counter = int(counter)
            frame_path = imaged_dir / f"imaged_{str(counter).zfill(4)}.png"

            # print(frame_path)
            assert frame_path.is_file()
            frame_path = str(frame_path)

            frame_id = self.find_frame(row_list=[[video_id, counter, frame_path]])
            if len(frame_id) == 0:
                # video_id INT NOT NULL,
                # count INT NOT NULL,
                # path VARCHAR (255) UNIQUE NOT NULL,
                self.insert_frame(row_list=[[video_id, counter, frame_path]])
                frame_id = self.find_frame(row_list=[[video_id, counter, frame_path]])
                if len(frame_id) == 0:
                    raise RuntimeError
            frame_id = frame_id[0]
            # print(f"frame id {frame_id}")

            _ = counter
            # print(f"counter         {counter}")
            # print(f"bbox            {bbox.shape}")
            # print(f"kps             {kps.shape}")
            # print(f"det_score       {det_score}")
            # print(f"landmark_3d_68  {landmark_3d_68.shape}")
            # print(f"pose            {pose.shape}")
            # print(f"landmark_2d_106 {landmark_2d_106.shape}")
            # print(f"gender          {gender}")
            # print(f"age             {age}")
            # print(f"embedding       {embedding.shape}")
            # print(row)

            row_inference = [
                frame_id,
                bbox,
                kps,
                det_score,
                landmark_3d_68,
                pose,
                landmark_2d_106,
                gender,
                age,
                embedding,
            ]
            self.insert_inference(row_list=[row_inference])
            inference_id = self.find_inference(row_list=[row_inference])
            if len(inference_id) == 0:
                raise RuntimeError
            inference_id = inference_id[0]
            # print(f"inference id {inference_id}")

    def parse_video_row(self, row: dict):
        counter = int(row["cnt"])
        bbox = np.array(
            literal_eval(
                re.sub("\s+", " ", row["bbox"].strip().replace("\n", ""))
                .replace("[ ", "[")
                .replace(" ", ",")
            )
        )
        kps = np.array(
            literal_eval(
                re.sub("\s+", " ", row["kps"].strip().replace("\n", ""))
                .replace("[ ", "[")
                .replace(" ", ",")
            )
        )
        det_score = float(row["det_score"])
        landmark_3d_68 = np.array(
            literal_eval(
                re.sub("\s+", " ", row["landmark_3d_68"].strip().replace("\n", ""))
                .replace("[ ", "[")
                .replace(" ", ",")
            )
        )
        pose = np.array(
            literal_eval(
                re.sub("\s+", " ", row["pose"].strip().replace("\n", ""))
                .replace("[ ", "[")
                .replace(" ", ",")
            )
        )
        landmark_2d_106 = np.array(
            literal_eval(
                re.sub("\s+", " ", row["landmark_2d_106"].strip().replace("\n", ""))
                .replace("[ ", "[")
                .replace(" ", ",")
            )
        )
        gender = int(row["gender"])
        age = int(row["age"])
        embedding = np.array(
            literal_eval(
                re.sub("\s+", " ", row["embedding"].strip().replace("\n", ""))
                .replace("[ ", "[")
                .replace(" ", ",")
            )
        )

        # string = json.dumps(lst)
        # lst = json.loads(string)

        return (
            counter,
            json.dumps(bbox.tolist()),
            json.dumps(kps.tolist()),
            det_score,
            json.dumps(landmark_3d_68.tolist()),
            json.dumps(pose.tolist()),
            json.dumps(landmark_2d_106.tolist()),
            gender,
            age,
            json.dumps(embedding.tolist()),
        )

    def insert_video(self, row_list: list):
        conn = connect_to_database(db_airflow)
        if conn:
            try:
                insert_data_video(conn, row_list)
            except Exception as e:
                print(f"Error interacting with {db_airflow['database']}: {e}")
            conn.close()
        else:
            print(f"No connection to {db_airflow['database']}.")

    def insert_frame(self, row_list: list):
        conn = connect_to_database(db_airflow)
        if conn:
            try:
                insert_data_frame(conn, row_list)
            except Exception as e:
                print(f"Error interacting with {db_airflow['database']}: {e}")
            conn.close()
        else:
            print(f"No connection to {db_airflow['database']}.")

    def insert_inference(self, row_list: list):
        conn = connect_to_database(db_airflow)
        if conn:
            try:
                insert_data_inference(conn, row_list)
            except Exception as e:
                print(f"Error interacting with {db_airflow['database']}: {e}")
            conn.close()
        else:
            print(f"No connection to {db_airflow['database']}.")

    def find_video(self, row_list: list):
        conn = connect_to_database(db_airflow)
        id_list = []
        if conn:
            try:
                id_list = find_id_video(conn, row_list)
            except Exception as e:
                print(f"Error interacting with {db_airflow['database']}: {e}")
            conn.close()
        else:
            print(f"No connection to {db_airflow['database']}.")
        return id_list

    def find_frame(self, row_list: list):
        conn = connect_to_database(db_airflow)
        id_list = []
        if conn:
            try:
                id_list = find_id_frame(conn, row_list)
            except Exception as e:
                print(f"Error interacting with {db_airflow['database']}: {e}")
            conn.close()
        else:
            print(f"No connection to {db_airflow['database']}.")
        return id_list

    def find_inference(self, row_list: list):
        conn = connect_to_database(db_airflow)
        id_list = []
        if conn:
            try:
                id_list = find_id_inference(conn, row_list)
            except Exception as e:
                print(f"Error interacting with {db_airflow['database']}: {e}")
            conn.close()
        else:
            print(f"No connection to {db_airflow['database']}.")
        return id_list


if __name__ == "__main__":
    do = ProcessData()
    u_user = os.getlogin()
    u_video_name = "inauguracion_metro_santiago"
    do.add_video_data_to_database(
        video_path=Path(
            f"/home/{u_user}/repos_git/airflow/insightface/videos/{u_video_name}.mp4"
        ),
        data_dir=Path(
            f"/home/{u_user}/repos_git/airflow/insightface/output/{u_video_name}"
        ),
    )
