from pathlib import Path

from airflow.face_detector.video.process import ProcessVideo

if __name__ == "__main__":

    videos_dir = Path("/home/toopazo/repos_git/airflow/videos")
    output_dir = Path("/home/toopazo/repos_git/airflow/output")

    assert videos_dir.is_dir()
    assert output_dir.is_dir()

    videos = [x for x in videos_dir.iterdir() if x.is_file()]
    for video_path in videos:
        print(f"video_path {video_path}")
        o_dir = output_dir / video_path.stem
        print(f"output_dir {o_dir}")
        pv = ProcessVideo(video_path, o_dir)
        pv.process(verbose=True)
