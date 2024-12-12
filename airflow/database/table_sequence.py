from copy import deepcopy

from airflow.database.db_config import connect_and_execute
from airflow.database.db_services import insert_data_sequence, get_data_sequence_by_name
from airflow.database.table_inference import Inference


class Sequence:
    def __init__(self):
        self.key_sequence_id = "sequence_id"
        self.key_frame_id = "frame_id"
        self.key_inference_id = "inference_id"

        self.__datad = {
            self.key_sequence_id: [],
            self.key_frame_id: [],
            self.key_inference_id: [],
        }

    def print_info(self):
        # print(f"  sequence_id list  {self.__datad[self.key_sequence_id]}")
        # print(f"  frame_id    list  {self.__datad[self.key_frame_id]}")
        # print(f"  inference_id list {self.__datad[self.key_inference_id]}")
        print(f"  sequence_id list  {len(self.__datad[self.key_sequence_id])}")
        print(f"  frame_id    list  {len(self.__datad[self.key_frame_id])}")
        print(f"  inference_id list {len(self.__datad[self.key_inference_id])}")

    def drop_data(self, i0: int, i1: int):
        # for i in range(i0, i1):
        #     self.__datad[self.key_sequence_id].pop(i)
        #     self.__datad[self.key_frame_id].pop(i)
        #     self.__datad[self.key_inference_id].pop(i)
        # del my_list[2:6]
        if i1 != 0:
            del self.__datad[self.key_sequence_id][i0:i1]
            del self.__datad[self.key_frame_id][i0:i1]
            del self.__datad[self.key_inference_id][i0:i1]
        else:
            # self.__datad[self.key_sequence_id].pop(i0)
            # self.__datad[self.key_frame_id].pop(i0)
            # self.__datad[self.key_inference_id].pop(i0)
            del self.__datad[self.key_sequence_id][i0:]
            del self.__datad[self.key_frame_id][i0:]
            del self.__datad[self.key_inference_id][i0:]
        self.verify_data_integrity()

    def initialize(self, infer: Inference, sequence_id: int):
        self.__datad = {
            self.key_sequence_id: [sequence_id],
            self.key_frame_id: [infer.frame_id],
            self.key_inference_id: [infer.inference_id],
        }
        self.verify_data_integrity()

    def append_inference(self, infer: Inference, sequence_id: int):
        self.__datad[self.key_sequence_id].append(sequence_id)
        self.__datad[self.key_frame_id].append(infer.frame_id)
        self.__datad[self.key_inference_id].append(infer.inference_id)

        self.verify_data_integrity()

    def verify_data_integrity(self):
        nk = len(self.__datad.keys())
        assert nk == 3
        lk1 = len(self.__datad[self.key_sequence_id])
        lk2 = len(self.__datad[self.key_frame_id])
        lk3 = len(self.__datad[self.key_inference_id])
        assert lk1 == lk2 == lk3

    def get_inference_id_list(self):
        return self.__datad[self.key_inference_id]

    def get_frame_id_list(self):
        return self.__datad[self.key_frame_id]

    def insert_into_database(self, sequence_name: str):
        if len(self.__datad[self.key_frame_id]) == 0:
            return

        for ix, frame_id in enumerate(self.__datad[self.key_frame_id]):
            inference_id = self.__datad[self.key_inference_id][ix]
            row_sequence = [frame_id, inference_id, sequence_name]
            connect_and_execute(
                service_fnct=insert_data_sequence,
                row_list=[row_sequence],
            )

        # row_name = [sequence_name]
        # data_list = connect_and_execute(
        #     service_fnct=find_data_sequence,
        #     row_list=[row_name],
        # )

        # data = data_list[0]
        # return data

    def load_from_database(self, sequence_name: str):
        row_name = [sequence_name]
        data_list = connect_and_execute(
            service_fnct=get_data_sequence_by_name,
            row_list=[row_name],
        )

        data = data_list[0]

        print(f"  There are {len(data)} inference in sequence {sequence_name}")

        self.__datad = {}
        for row in data:
            assert len(row) == 4
            sequence_id = row[0]
            frame_id = row[1]
            inference_id = row[2]
            # sequence_name = row[3]

            infer = Inference()
            infer.frame_id = frame_id
            infer.inference_id = inference_id

            # I do not need to load the whole row in order to append it to Sequence
            # infer.load_from_database(inference_id=inference_id)

            # print()
            # print(f"  sequence_id   {sequence_id}")
            # print(f"  frame_id      {frame_id}")
            # print(f"  inference_id  {inference_id}")
            # print(f"  sequence_name {sequence_name}")

            # exit(0)
            # inference_list.append(deepcopy(infer))

            if len(self.__datad.keys()) == 0:
                self.initialize(deepcopy(infer), sequence_id)
            else:
                self.append_inference(deepcopy(infer), sequence_id)
