"""
Script principal del detector de rostros
"""

import os
import sys
from pathlib import Path

from airflow.face_detector.video.process import ProcessVideo

if __name__ == "__main__":
    videos_dir = Path(sys.argv[1]).absolute()
    output_dir = Path(sys.argv[2]).absolute()

    print(f"User input videos_dir    {videos_dir}")
    print(f"User input output_dir    {output_dir}")

    assert videos_dir.is_dir()
    assert output_dir.is_dir()

    videos = [x for x in videos_dir.iterdir() if x.is_file()]
    for video_path in videos:
        print(f"video_path {video_path}")
        o_dir = output_dir / video_path.stem
        print(f"output_dir {o_dir}")
        pv = ProcessVideo(video_path, o_dir)
        pv.process(verbose=True)

    # python -m airflow.face_detector.process_video \
    #   /home/${USER}/repos_git/airflow/videos /home/${USER}/repos_git/airflow/output
