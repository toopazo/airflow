import os
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from videoreader import VideoReader
from src.frame.process import ProcessFrame


class ProcessVideo:
    def __init__(self, video_path: Path, output_dir: Path):
        self.video_reader = VideoReader(video_path)
        self.output_dir = output_dir
        self.video_name = video_path.stem

        self.pf = ProcessFrame()
        self.datad_name = "name"
        self.datad_cnt = "cnt"
        self.datad = {self.datad_name: [], self.datad_cnt: []}
        for k in self.pf.key_list:
            self.datad[k] = []

    def process(self, verbose: bool):

        cnt = 0
        digits = len(str(len(self.video_reader)))

        if verbose:
            imaged_dir = self.output_dir / "imaged"
            os.makedirs(imaged_dir, exist_ok=True)

        for frame in tqdm(self.video_reader, total=self.video_reader.number_of_frames):
            image = self.pf.cv2_to_pil(frame)
            faces = self.pf.inference_pil(image)

            self.faces_to_database(faces, cnt)

            if verbose:
                imaged = self.pf.draw_faces_pil(image, faces)
                imaged_path = imaged_dir / f"imaged_{str(cnt).zfill(digits)}.png"
                imaged.save(imaged_path)

            cnt = cnt + 1

            # if cnt == 50:
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

    def faces_to_database(self, faces: list, cnt: int):
        for face in faces:
            self.datad[self.datad_name].append(self.video_name)
            self.datad[self.datad_cnt].append(cnt)
            for k in self.pf.key_list:
                self.datad[k] = str(face[k])

    def make_dataframe(self, output_dir: Path):
        df = pd.DataFrame(self.datad)
        df.to_csv(output_dir / "dataframe.csv")
