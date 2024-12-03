import os
import re
from ast import literal_eval
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm
import pandas as pd

from airflow.database.db_config import connect_and_execute
from airflow.database.db_services import (
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

        # print(video_name)
        # print(df)
        # print(df[df["cnt"] <= 2])
        # return

        # name VARCHAR (255) UNIQUE NOT NULL,
        # path VARCHAR (255) UNIQUE NOT NULL,

        connect_and_execute(
            service_fnct=insert_data_video,
            row_list=[[str(video_name), str(video_path)]],
        )
        video_id = connect_and_execute(
            service_fnct=find_id_video,
            row_list=[[str(video_name), str(video_path)]],
        )
        if len(video_id) == 0:
            raise RuntimeError
        video_id = video_id[0]
        # print(f"video id {video_id}")

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
            _frame_path = imaged_dir / f"imaged_{str(counter).zfill(4)}.png"

            # print(frame_path)
            assert _frame_path.is_file()
            frame_path = str(_frame_path)

            # print([video_id, counter, frame_path])
            # continue

            # frame_id = self.find_frame(row_list=[[video_id, counter, frame_path]])
            frame_id = connect_and_execute(
                service_fnct=find_id_frame,
                row_list=[[video_id, counter, frame_path]],
            )
            if len(frame_id) == 0:
                # video_id INT NOT NULL,
                # count INT NOT NULL,
                # path VARCHAR (255) UNIQUE NOT NULL,
                # self.insert_frame(row_list=[[video_id, counter, frame_path]])
                connect_and_execute(
                    service_fnct=insert_data_frame,
                    row_list=[[video_id, counter, frame_path]],
                )
                # frame_id = self.find_frame(row_list=[[video_id, counter, frame_path]])
                frame_id = connect_and_execute(
                    service_fnct=find_id_frame,
                    row_list=[[video_id, counter, frame_path]],
                )
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
            # self.insert_inference(row_list=[row_inference])
            connect_and_execute(
                service_fnct=insert_data_inference,
                row_list=[row_inference],
            )
            # inference_id = self.find_inference(row_list=[row_inference])
            inference_id = connect_and_execute(
                service_fnct=find_id_inference,
                row_list=[row_inference],
            )
            if len(inference_id) == 0:
                raise RuntimeError
            inference_id = inference_id[0]
            # print(f"inference id {inference_id}")

            # if counter > 1:
            #     return

    def parse_video_row(self, row: dict):
        counter = int(row["cnt"])
        bbox = np.array(
            literal_eval(
                re.sub(r"\s+", " ", row["bbox"].strip().replace("\n", ""))
                .replace("[ ", "[")
                .replace(" ", ",")
            )
        )
        kps = np.array(
            literal_eval(
                re.sub(r"\s+", " ", row["kps"].strip().replace("\n", ""))
                .replace("[ ", "[")
                .replace(" ", ",")
            )
        )
        det_score = float(row["det_score"])
        landmark_3d_68 = np.array(
            literal_eval(
                re.sub(r"\s+", " ", row["landmark_3d_68"].strip().replace("\n", ""))
                .replace("[ ", "[")
                .replace(" ", ",")
            )
        )
        pose = np.array(
            literal_eval(
                re.sub(r"\s+", " ", row["pose"].strip().replace("\n", ""))
                .replace("[ ", "[")
                .replace(" ", ",")
            )
        )
        landmark_2d_106 = np.array(
            literal_eval(
                re.sub(r"\s+", " ", row["landmark_2d_106"].strip().replace("\n", ""))
                .replace("[ ", "[")
                .replace(" ", ",")
            )
        )
        gender = int(row["gender"])
        age = int(row["age"])
        embedding = np.array(
            literal_eval(
                re.sub(r"\s+", " ", row["embedding"].strip().replace("\n", ""))
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


if __name__ == "__main__":
    do = ProcessData()
    u_user = os.getlogin()
    U_VIDEO_NAME = "inauguracion_metro_santiago"
    do.add_video_data_to_database(
        video_path=Path(f"/home/{u_user}/repos_git/airflow/videos/{U_VIDEO_NAME}.mp4"),
        data_dir=Path(f"/home/{u_user}/repos_git/airflow/output/{U_VIDEO_NAME}"),
    )