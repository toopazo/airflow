"""
Script para pasar de detecciones a secuencias
"""

import os
from pathlib import Path
from copy import deepcopy

from airflow.utils.video_handler import VideoHandler
from airflow.utils.image_painter import ImagePainter
from airflow.database.table_inference import Inference
from airflow.face_sequencer.filter_geometry import GeometryFilter
from airflow.face_sequencer.filter_dynamics import DynamicFilter


class SequenceFinder:
    def __init__(self, video_id: int):
        self.video_id = video_id
        self.vidhan = VideoHandler(video_id=video_id)

    # def show_sequence(self):
    #     imgptr = ImagePainter(self.video_id)
    #     imgptr.show_frames(frame_i=1, frame_j=100)

    def find_sequences(self):
        frame_i = 1
        frame_j = 3
        infer_dict = self.vidhan.get_data_inferences(frame_i=frame_i, frame_j=frame_j)

        # imgptr = ImagePainter(self.video_id)
        # image_list = imgptr.auto_draw_annotations(frame_i=frame_i, frame_j=frame_j)

        geom_infer_dict = self.filter_by_geometry(infer_dict=infer_dict)
        dyn_infer_dict = self.filter_by_dynamics(infer_dict=geom_infer_dict)

    def filter_by_geometry(self, infer_dict: dict):
        # Durante una misma secuencia
        #   El tamaño de las caras de ser grande
        #   Las caras deben mirar de frente (kpi con criterio geometrico)
        #   La distancia entre caras debe ser alta (iou == 0)

        new_infer_dict = {}
        for k, v in infer_dict.items():
            infer_list = v["infer_list"]
            frame_id = k

            print(f"There are {len(infer_list)} inferences in frame_id {frame_id}")
            # print(f"infer_dict[{k}][infer_list] has len {len(infer_list)}")

            for infer in infer_list:
                assert isinstance(infer, Inference)
                gf = GeometryFilter(infer)
                approved = gf.apply()

                infer_id = infer.inference_id
                if approved:
                    new_infer_dict[k] = deepcopy(v)
                    print(f"  inference_id {infer_id} accepted by geometry")
                else:
                    print(f"  inference_id {infer_id} rejected by geometry")

        return new_infer_dict

    def filter_by_dynamics(self, infer_dict: dict) -> dict:
        # Durante una misma secuencia
        #   La velocidad de cada cara debe ser baja
        #   La trayectoria de cada cara debe ser identificable de manera simple (las oclusiónes
        # implican el fin de la secuencia)

        # El resultado de este filtro es un tracking
        # new_infer_dict = deepcopy(infer_dict)
        # face_ids = deepcopy(new_infer_dict)
        # return face_ids

        list_of_keys = list(infer_dict.keys())
        print(f"list_of_keys {list_of_keys}")

        v = infer_dict[list_of_keys[0]]
        infer_list: list[Inference] = v["infer_list"]
        tracking: list[DynamicFilter] = []
        for infer in infer_list:
            dynfil = DynamicFilter()
            dynfil.initialize(infer=infer)
            tracking.append(dynfil)

        list_of_keys.pop(0)

        new_infer_dict = {}
        for k in list_of_keys:
            v = infer_dict[k]
            infer_list = v["infer_list"]
            frame_id = k

            print(f"There are {len(infer_list)} inferences in frame_id {frame_id}")
            # print(f"infer_dict[{k}][infer_list] has len {len(infer_list)}")

            left_over = [e.inference_id for e in infer_list]
            for dynfil in tracking:
                approved = dynfil.apply(infer_list=infer_list)

                if approved:
                    new_infer_dict[k] = deepcopy(v)
                    print(f"  The sequence {dynfil.sequence} was accepted by dynamics")
                    left_over.remove(dynfil.sequence[-1])
                else:
                    print(
                        f"  The sequence {dynfil.sequence} was rejected (interrupted) by dynamics"
                    )
            print(f"left_over {left_over}")

            if len(left_over) > 1:
                raise RuntimeError

        return new_infer_dict


if __name__ == "__main__":
    v1_handler = SequenceFinder(video_id=1)
    v1_handler.find_sequences()
    # v1_handler.show_sequence()
