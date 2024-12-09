"""
Script para interactuar con la db
"""

from pathlib import Path
from PIL import Image

from airflow.database.db_config import connect_and_execute
from airflow.database.table_inference import Inference


class VideoHandler:
    """
    Clase para interactuar con la base de datos
    handler, broker, dealer, operator,
    shipper, merchant, trader, trafficker
    """

    def __init__(self, video_id: int):
        self.video_id = video_id

        video_row = self.get_data_video()
        # (1, 'inauguracion_metro_santiago', '/home/ .. /airflow/videos/
        # inauguracion_metro_santiago.mp4', datetime.datetime(2024, 12, 4, 2, 44, 52, 677154))
        self.video_name = str(video_row[1])
        self.video_path = Path(video_row[2])
        assert self.video_path.is_file()

        row = [self.video_id]
        res_list = connect_and_execute(
            service_fnct=self.find_id_frame,
            row_list=[row],
        )
        frame_ids = res_list[0]
        self.frame_ids = frame_ids

        row = [self.video_id]
        res_list = connect_and_execute(
            service_fnct=self.find_row_frame,
            row_list=[row],
        )
        frame_rows = res_list[0]
        self.frame_rows = frame_rows

    def parse_inference_rows(self, inference_rows: list):
        infer_list = []
        for i, row in enumerate(inference_rows):
            _ = i
            row = list(row)
            infer = Inference()
            infer.parse_row(row)
            # print(f"{i}-th inference")
            # infer.print_info()
            infer_list.append(infer)
        return infer_list

    def find_row_video(self, connection, row_list: list):
        with connection.cursor() as cursor:
            res_list = []
            for row in row_list:
                query = """
                    SELECT * FROM video
                    WHERE
                        id = %s;
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

    def find_row_frame(self, connection, row_list: list):
        with connection.cursor() as cursor:
            res_list = []
            for row in row_list:
                query = """
                    SELECT * FROM frame
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
                # if len(res) > 0:
                #     # ids = res
                #     ids = [int(e[0]) for e in res]
                #     res_list.append(ids)
                res_list.append(res)
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

    def get_data_video(self):
        row = [self.video_id]
        res_list = connect_and_execute(
            service_fnct=self.find_row_video,
            row_list=[row],
        )
        video_rows = res_list[0]
        video_row = video_rows[0]
        return video_row

    def get_data_frames(self, frame_i: int, frame_j: int) -> dict:
        assert frame_i < frame_j

        # row = [self.video_id]
        # res_list = connect_and_execute(
        #     service_fnct=self.find_row_frame,
        #     row_list=[row],
        # )
        # frame_rows = res_list[0]
        frame_rows = self.frame_rows

        # print(f"There are {len(frame_rows)} frames in video_id {self.video_id}")
        # print(f"  frame_ids len {len(frame_rows)}")
        # print(f"  frame_ids[0]  {frame_rows[0]}")
        # print(f"  frame_ids[1]  {frame_rows[1]}")
        # print(f"  frame_ids[-1] {frame_rows[-1]}")
        # print(ids)

        # ids = list of all frame_id that point to video_id

        frame_dict = {}
        for frame_row in frame_rows:
            # frame_row = list(frame_row)
            frame_id = frame_row[0]
            frame_path = frame_row[3]
            # print(frame_id, frame_path)

            if frame_id < frame_i:
                continue
            if frame_id > frame_j - 1:
                continue

            frame_path = Path(frame_path)
            assert frame_path.is_file()
            image = Image.open(frame_path)

            frame_dict[frame_id] = {"image": image}

        return frame_dict

    def get_data_inferences(self, frame_i: int, frame_j: int):
        assert frame_i < frame_j

        # row = [self.video_id]
        # res_list = connect_and_execute(
        #     service_fnct=self.find_id_frame,
        #     row_list=[row],
        # )
        # frame_ids = res_list[0]
        frame_ids = self.frame_ids

        # print(f"There are {len(frame_ids)} frames in video_id {self.video_id}")
        # print(f"  frame_ids len {len(frame_ids)}")
        # print(f"  frame_ids[0]  {frame_ids[0]}")
        # print(f"  frame_ids[1]  {frame_ids[1]}")
        # print(f"  frame_ids[-1] {frame_ids[-1]}")
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
            # print(f"There are {len(inference_rows)} inferences in frame_id {frame_id}")

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
    v1_handler = VideoHandler(1)
    inferd = v1_handler.get_data_inferences(frame_i=1, frame_j=150)
