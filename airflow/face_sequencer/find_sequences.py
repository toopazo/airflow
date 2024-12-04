"""
Script para pasar de detecciones a secuencias
"""

import os
from pathlib import Path
from copy import deepcopy
import numpy as np

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
        frame_j = 100
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

            # print(f"There are {len(infer_list)} inferences in frame_id {frame_id}")
            # print(f"infer_dict[{k}][infer_list] has len {len(infer_list)}")

            new_infer_list = []
            for infer in infer_list:
                assert isinstance(infer, Inference)
                gf = GeometryFilter(infer)
                approved = gf.apply()

                infer_id = infer.inference_id
                if approved:
                    new_infer_list.append(deepcopy(infer))
                    print(f"  inference_id {infer_id} accepted by geometry")
                else:
                    print(f"  inference_id {infer_id} rejected by geometry")

            new_infer_dict[k] = {"infer_list": new_infer_list}

            print(
                f"There are {len(infer_list)} inferences in frame_id {frame_id} ({len(new_infer_list)} were accepted)"
            )

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

            left_overs = [e.inference_id for e in infer_list]
            terminated_seqs = []
            for ix, dynfil in enumerate(tracking):
                approved = dynfil.apply(infer_list=infer_list)

                if approved:
                    new_infer_dict[k] = deepcopy(v)
                    seq_len = len(dynfil.sequence)
                    print(
                        f"  The sequence with len {seq_len} was accepted (continued) by dynamics"
                    )
                    left_overs.remove(dynfil.sequence[-1])
                else:
                    seq_len = len(dynfil.sequence)
                    print(
                        f"  The sequence with len {seq_len} was rejected (interrupted) by dynamics"
                    )
                    terminated_seqs.append(ix)
                    # raise RuntimeError

            # Handle terminated sequences
            if len(terminated_seqs) > 0:
                print(f"  Handling terminated_seqs {terminated_seqs} -------->")
                for rm_ix in terminated_seqs:
                    rm_infer = tracking.pop(rm_ix)
                    print(
                        f"  Removing sequence terminated at inference_id {rm_infer.sequence[-1]}"
                    )

            # Handle left overs
            if len(left_overs) > 0:
                # raise RuntimeError
                print(f"  Handling left_overs {left_overs} -------->")
                for left_over_id in left_overs:
                    for infer in infer_list:
                        if infer.inference_id == left_over_id:
                            print(
                                f"  A new sequence was started from inference_id {left_over_id}"
                            )
                            dynfil = DynamicFilter()
                            dynfil.initialize(infer=infer)
                            tracking.append(dynfil)

        return new_infer_dict


if __name__ == "__main__":
    v1_handler = SequenceFinder(video_id=1)
    v1_handler.find_sequences()
    # v1_handler.show_sequence()
