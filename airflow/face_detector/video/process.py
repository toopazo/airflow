import os
from copy import deepcopy
from pathlib import Path
import numpy as np
from tqdm import tqdm

import pandas as pd
from videoreader import VideoReader

from airflow.face_detector.frame.process import ProcessFrame
from airflow.database.table_inference import Inference


class ProcessVideo:
    def __init__(self, video_path: Path, output_dir: Path):
        self.video_reader = VideoReader(video_path)
        self.output_dir = output_dir
        self.video_name = video_path.stem

        self.pf = ProcessFrame()
        self.datad_name = "name"
        self.datad_counter = "counter"
        self.datad = {self.datad_name: [], self.datad_counter: []}
        for k in self.pf.key_list:
            self.datad[k] = []

    def process(self, verbose: bool):

        counter = 1
        digits = len(str(len(self.video_reader)))

        if verbose:
            image_dir = self.output_dir / "image"
            os.makedirs(image_dir, exist_ok=True)
            imaged_dir = self.output_dir / "imaged"
            os.makedirs(imaged_dir, exist_ok=True)

        for frame in tqdm(self.video_reader, total=self.video_reader.number_of_frames):
            image = self.pf.cv2_to_pil(frame)
            faces = self.pf.inference_pil(image)

            self.faces_to_datadict(faces, counter)

            if verbose:
                image_path = image_dir / f"image_{str(counter).zfill(digits)}.png"
                image.save(image_path)

                imaged = self.pf.draw_faces_pil(image, faces)
                imaged_path = imaged_dir / f"imaged_{str(counter).zfill(digits)}.png"
                imaged.save(imaged_path)

            counter = counter + 1

            # if counter == 50:
            #     break

        if verbose:
            self.make_video(imaged_dir)
            self.make_dataframe(imaged_dir.parent)

    def make_video(self, input_dir: Path):
        images = [x for x in input_dir.iterdir() if x.is_file()]
        images = sorted(images)

        fps = self.video_reader.frame_rate
        in_pattern = input_dir / "*.png"
        out_path = input_dir.parent / "imaged.mp4"  # f"{self.video_name}.mp4"
        cmd = f"ffmpeg -framerate {fps} -pattern_type glob -i '{in_pattern}' -c:v libx264 -pix_fmt yuv420p {out_path}"
        print(cmd)
        os.system(cmd)

    def faces_to_datadict(self, faces: list, counter: int):
        # faces[0] keys dict_keys(['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'])
        for face in faces:
            print(f"counter {counter} face {face['bbox']}")
            self.datad[self.datad_name].append(self.video_name)
            self.datad[self.datad_counter].append(counter)
            for k in self.pf.key_list:
                v = face[k]
                if isinstance(v, np.ndarray):
                    v = v.tolist()
                self.datad[k].append(v)

            # infer = Inference()
            # row = [
            #     0,  # inference_id
            #     0,  # frame_id
            #     face["bbox"],
            #     face["kps"],
            #     face["det_score"],
            #     face["landmark_3d_68"],
            #     face["pose"],
            #     face["landmark_2d_106"],
            #     face["gender"],
            #     face["age"],
            #     face["embedding"],
            # ]
            # infer.parse_row(row=row)
            # self.datad[self.pf.key_bbox] = infer.bbox.tolist()
            # self.datad[self.pf.key_kps] = infer.kps.tolist()
            # self.datad[self.pf.key_det_score] = infer.det_score
            # self.datad[self.pf.key_landmark_3d_68] = infer.landmark_3d_68.tolist()
            # self.datad[self.pf.key_pose] = infer.pose.tolist()
            # self.datad[self.pf.key_landmark_2d_106] = infer.landmark_2d_106.tolist()
            # self.datad[self.pf.key_gender] = infer.gender
            # self.datad[self.pf.key_age] = infer.age
            # self.datad[self.pf.key_embedding] = infer.embedding.tolist()

    def make_dataframe(self, output_dir: Path):
        df = pd.DataFrame(self.datad)
        df.to_csv(output_dir / "dataframe.csv")
