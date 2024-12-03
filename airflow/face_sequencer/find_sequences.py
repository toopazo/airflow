"""
Script para pasar de detecciones a secuencias
"""

import os
from pathlib import Path
from copy import deepcopy

from airflow.face_sequencer.db_handler import DbHandler


class SequenceFinder:
    def __init__(self, video_id: int):
        self.video_handler = DbHandler(video_id=video_id)

    def find_sequences(self):
        inferd = self.video_handler.get_inferences(frame_i=1, frame_j=150)

        # inferd = self.filter_by_geometry(inferd=inferd)
        # inferd = self.filter_by_dynamics(inferd=inferd)

    def filter_by_geometry(self, inferd: dict):
        new_inferd = deepcopy(inferd)
        # Durante una misma secuencia
        #   El tamaño de las caras de ser grande
        #   Las caras deben mirar de frente (kpi con criterio geometrico)
        #   La distancia entre caras debe ser alta (iou == 0)
        return new_inferd

    def filter_by_dynamics(self, inferd: dict) -> dict:
        new_inferd = deepcopy(inferd)
        # Durante una misma secuencia
        #   La velocidad de cada cara debe ser baja
        #   La trayectoria de cada cara debe ser identificable de manera simple (las oclusiónes
        # implican el fin de la secuencia)

        # El resultado de este filtro es un tracking
        face_ids = deepcopy(new_inferd)
        return face_ids


if __name__ == "__main__":
    v1_handler = SequenceFinder(video_id=1)
    v1_handler.find_sequences()
