"""
Script para interactuar con la db
"""

import os
from pathlib import Path
import json
import numpy as np
from copy import deepcopy

from airflow.database.db_config import connect_and_execute


class RowInference:
    """
    Clase que corresponde a una fila de la tabla inference de la base de datos
    """

    def __init__(self, row: list):
        if len(row) == 11:
            for i, e in enumerate(row):
                # i_str = str(i).rjust(2)
                if isinstance(e, int) or isinstance(e, float):
                    # e_len = "1".ljust(10)
                    # e_type = str(type(e)).ljust(25)
                    # e_shape = "(1,)".ljust(10)
                    pass
                else:
                    # e_len = str(len(e)).ljust(10)
                    e = np.array(json.loads(e))
                    # e_type = str(type(e)).ljust(25)
                    # e_shape = str(e.shape).ljust(10)
                # print(f"  row[{i_str}] type {e_type} shape {e_shape}")

                if i == 0:
                    self.inference_id = deepcopy(e)
                elif i == 1:
                    self.frame_id = deepcopy(e)
                elif i == 2:
                    self.bbox = deepcopy(e)
                elif i == 3:
                    self.kps = deepcopy(e)
                elif i == 4:
                    self.det_score = deepcopy(e)
                elif i == 5:
                    self.landmark_3d_68 = deepcopy(e)
                elif i == 6:
                    self.pose = deepcopy(e)
                elif i == 7:
                    self.landmark_2d_106 = deepcopy(e)
                elif i == 8:
                    self.gender = deepcopy(e)
                elif i == 9:
                    self.age = deepcopy(e)
                elif i == 10:
                    self.embedding = deepcopy(e)
                else:
                    raise IndexError
        else:
            raise IndexError

    def print_info(self):
        print(f"  inference_id    value {self.inference_id}")
        print(f"  frame_id        value {self.frame_id}")
        print(f"  bbox            shape {self.bbox.shape}")
        print(f"  kps             shape {self.kps.shape}")
        print(f"  det_score       value {self.det_score}")
        print(f"  landmark_3d_68  shape {self.landmark_3d_68.shape}")
        print(f"  pose            shape {self.pose.shape}")
        print(f"  landmark_2d_106 shape {self.landmark_2d_106.shape}")
        print(f"  gender          value {self.gender}")
        print(f"  age             value {self.age}")
        print(f"  embedding       shape {self.embedding.shape}")


class DbHandler:
    """
    Clase para interactuar con la base de datos
    handler, broker, dealer, operator,
    shipper, merchant, trader, trafficker
    """

    def __init__(self, video_id: int):
        self.video_id = video_id

    def parse_inference_rows(self, inference_rows: list):
        infer_list = []
        for i, row in enumerate(inference_rows):
            row = list(row)
            infer = RowInference(row)
            # print(f"{i}-th inference")
            # infer.print_info()
            infer_list.append(infer)
        return infer_list

    def find_id_frame(self, connection, row_list: list):
        with connection.cursor() as cursor:
            res_list = []
            for row in row_list:
                query = """
                    SELECT id FROM frame
                    WHERE
                        video_id = %s;
                """
                cursor.execute(query, row)
                res = cursor.fetchall()
                # print(f"res len {len(res)}")
                # print(f"res[0]  {res[0]}")
                # print(f"res[1]  {res[1]}")
                # print(f"res[-1] {res[-1]}")
                # [(1,)]
                if len(res) > 0:
                    # ids = res
                    ids = [int(e[0]) for e in res]
                    res_list.append(ids)
                # res_list.append(res)
        return res_list

    def find_row_inference(self, connection, row_list: list):
        with connection.cursor() as cursor:
            res_list = []
            for row in row_list:
                query = """
                    SELECT * FROM inference
                    WHERE
                        frame_id = %s;
                """
                cursor.execute(query, row)
                res = cursor.fetchall()
                # print(f"res len {len(res)}")
                # print(f"res[0]  {res[0]}")
                # print(f"res[1]  {res[1]}")
                # print(f"res[-1] {res[-1]}")
                # [(1,)]
                # if len(res) > 0:
                #     # ids = res
                #     ids = [int(e[0]) for e in res]
                #     res_list.append(ids)
                res_list.append(res)
        return res_list

    def get_inferences(self, frame_i: int, frame_j: int):
        assert frame_i < frame_j

        row = [self.video_id]
        res_list = connect_and_execute(
            service_fnct=self.find_id_frame,
            row_list=[row],
        )
        frame_ids = res_list[0]

        print(f"There are {len(frame_ids)} frames in video_id {self.video_id}")
        print(f"  frame_ids len {len(frame_ids)}")
        print(f"  frame_ids[0]  {frame_ids[0]}")
        print(f"  frame_ids[1]  {frame_ids[1]}")
        print(f"  frame_ids[-1] {frame_ids[-1]}")
        # print(ids)

        # ids = list of all frame_id that point to video_id

        infer_dict = {}
        for frame_id in frame_ids:

            if frame_id < frame_i:
                continue
            if frame_id > frame_j - 1:
                continue

            row = [frame_id]
            res_list = connect_and_execute(
                service_fnct=self.find_row_inference,
                row_list=[row],
            )
            inference_rows = res_list[0]

            # res_list = list of all inference_row that point to video_id

            # print()
            print(f"There are {len(inference_rows)} inferences in frame_id {frame_id}")

            infer_list = self.parse_inference_rows(inference_rows)

            infer_dict[frame_id] = {"infer_list": infer_list}

            # if frame_id > 10:
            #     return

            # inference = inference_rows[0]
            # print()
            # print(f"inference len {len(inference)}")
            # print(f"inference[0]  {inference[0]}")
            # print(f"inference[1]  {inference[1]}")
            # print(f"inference[-1] {inference[-1]}")
            # print(inference)
        return infer_dict


if __name__ == "__main__":
    v1_handler = DbHandler(1)
    inferd = v1_handler.get_inferences(frame_i=1, frame_j=150)
