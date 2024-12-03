"""
Script para pasar de detecciones a secuencias
"""

import os
from pathlib import Path

from airflow.face_sequencer.db_handler import DbHandler


class SequenceFinder:
    def __init__(self, video_id: int):
        self.video_handler = DbHandler(video_id=1)

    def find_sequences(self):
        inferd = self.video_handler.get_inferences(frame_i=1, frame_j=150)


if __name__ == "__main__":
    v1_handler = SequenceFinder(video_id=1)
    v1_handler.find_sequences()
