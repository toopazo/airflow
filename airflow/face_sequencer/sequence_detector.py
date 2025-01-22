"""
Script para pasar de detecciones a secuencias
"""

import os
import sys
from pathlib import Path
from copy import deepcopy

from airflow.utils.video_handler import VideoHandler
from airflow.utils.image_painter import ImagePainter
from airflow.database.table_inference import Inference
from airflow.face_sequencer.filter_geometry import GeometryFilter
from airflow.face_sequencer.filter_dynamics import SequenceTracker
from airflow.database.table_sequence import Sequence


class SequenceDetector:
    def __init__(self, video_id: int, min_sequence_len: int, output_dir: Path):
        self.video_id = video_id
        self.min_sequence_len = min_sequence_len

        self.vid_han = VideoHandler(video_id=video_id)
        self.output_dir = output_dir / self.vid_han.video_name / "sequence"
        os.makedirs(self.output_dir, exist_ok=True)

        self.img_ptr = ImagePainter(video_id=self.video_id)

    def detect_sequences(self, frame_i: int, frame_j: int):
        infer_dict = self.vid_han.get_data_inferences(frame_i=frame_i, frame_j=frame_j)

        geom_infer_dict = self.filter_by_geometry(infer_dict=infer_dict)
        self.filter_by_dynamics(infer_dict=geom_infer_dict)

    def filter_by_geometry(self, infer_dict: dict):
        """
        Durante una misma secuencia
          El tamaño de las caras de ser grande
          Las caras deben mirar de frente (kpi con criterio geométrico)
          La distancia entre caras debe ser alta (iou == 0)
        """

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

    def filter_by_dynamics(self, infer_dict: dict):
        """
        Durante una misma secuencia
          La velocidad de cada cara debe ser baja
          La trayectoria de cada cara debe ser identificable de manera simple (la oclusión
          o cercanía de otros bbox implican el fin de la secuencia)

        El resultado de este filtro es un tracking
          new_infer_dict = deepcopy(infer_dict)
          face_ids = deepcopy(new_infer_dict)
          return face_ids
        """

        frame_id_list = list(infer_dict.keys())
        print(f"frame_id_list {frame_id_list}")

        v = infer_dict[frame_id_list[0]]
        infer_list: list[Inference] = v["infer_list"]
        seq_track = SequenceTracker(infer_list)

        frame_id_list.pop(0)

        for frame_id in frame_id_list:
            v = infer_dict[frame_id]
            infer_list = v["infer_list"]

            print(f"There are {len(infer_list)} inferences in frame_id {frame_id}")
            # print(f"infer_dict[{k}][infer_list] has len {len(infer_list)}")

            active_seq_list, terminated_seq_list = seq_track.update(infer_list)

            self.__save_terminated_sequence(terminated_seq_list)
            self.__insert_terminated_sequence(terminated_seq_list)

        # return .active_seq_list, terminated_seq_list

    def __save_terminated_sequence(self, terminated_seq_list: list[Sequence]):
        for ix, sequence in enumerate(terminated_seq_list):
            assert isinstance(sequence, Sequence)

            seq_frame_id = sequence.get_frame_id_list()
            seq_inference_id = sequence.get_inference_id_list()

            if len(seq_frame_id) < self.min_sequence_len:
                print()
                print(
                    f"Terminated sequence length {len(seq_frame_id)} < {self.min_sequence_len} "
                )
                print()
                continue

            last_frame_id = seq_frame_id[-1]
            frame_id_str = str(last_frame_id).zfill(6)
            ix_str = str(ix).zfill(6)
            sequence_name = f"frame_id_{frame_id_str}_terminated_sequence_{ix_str}"

            seq_bbox = []
            for infer_id in seq_inference_id:
                infer = Inference()
                infer.load_from_database(inference_id=infer_id)
                seq_bbox.append(infer.bbox)

            seq_path = self.output_dir / f"{sequence_name}.png"
            self.img_ptr.draw_crop_list(
                bbox_list=seq_bbox,
                frame_id_list=seq_frame_id,
                image_path=seq_path,
            )

    def __insert_terminated_sequence(self, terminated_seq_list: list[Sequence]):

        for ix, sequence in enumerate(terminated_seq_list):
            assert isinstance(sequence, Sequence)

            seq_frame_id = sequence.get_frame_id_list()

            if len(seq_frame_id) < self.min_sequence_len:
                print()
                print(
                    f"Terminated sequence length {len(seq_frame_id)} < {self.min_sequence_len} "
                )
                print()
                continue

            last_frame_id = seq_frame_id[-1]
            frame_id_str = str(last_frame_id).zfill(6)
            ix_str = str(ix).zfill(6)
            sequence_name = f"frame_id_{frame_id_str}_terminated_sequence_{ix_str}"

            print()
            print(f"Terminated sequence {sequence_name} will be  inserted into the db")
            print()
            sequence.insert_into_database(sequence_name=sequence_name)


if __name__ == "__main__":
    u_video_id = int(sys.argv[1])
    u_output_dir = Path(sys.argv[2]).absolute()

    print(f"User input video_id      {u_video_id}")
    print(f"User input output_dir    {u_output_dir}")

    assert u_output_dir.is_dir()

    seq_det = SequenceDetector(
        video_id=u_video_id, min_sequence_len=5, output_dir=u_output_dir
    )
    # seq_det.detect_sequences(frame_i=1, frame_j=100,save=False)
    # seq_det.detect_sequences(frame_i=250, frame_j=322, save=True)
    seq_det.detect_sequences(frame_i=250, frame_j=500)
