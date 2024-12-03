from pathlib import Path
from src.video.process import ProcessVideo

if __name__ == "__main__":
    videos_dir = Path(__file__).parent / "videos"
    output_dir = Path(__file__).parent / "output"

    videos = [x for x in videos_dir.iterdir() if x.is_file()]
    for video_path in videos:
        print(f"video_path {video_path}")
        o_dir = output_dir / video_path.stem
        print(f"output_dir {o_dir}")
        pv = ProcessVideo(video_path, o_dir)
        pv.process(verbose=True)
