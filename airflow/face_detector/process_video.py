"""
Script principal del detector de rostros
"""

import os
import sys
from pathlib import Path

from airflow.face_detector.video.process import ProcessVideo

if __name__ == "__main__":
    video_path = Path(sys.argv[1]).absolute()
    output_dir = Path(sys.argv[2]).absolute()

    print(f"User input video_path    {video_path}")
    print(f"User input output_dir    {output_dir}")

    assert video_path.is_file()
    assert output_dir.is_dir()

    # print(f"video_path {video_path}")
    o_dir = output_dir / video_path.stem
    # print(f"output_dir {o_dir}")
    pv = ProcessVideo(video_path, o_dir)
    pv.process(verbose=True)
