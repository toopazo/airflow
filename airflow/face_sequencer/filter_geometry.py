"""
Script para filtrar caras por geometria
"""

# import os
# from pathlib import Path
from airflow.database.table_inference import Inference


class GeometryFilter:
    def __init__(self, infer: Inference):
        self.infer = infer
        self.verbose = False
        self.min_area_px = 100 * 100

    def apply(self):
        x1, y1, x2, y2 = self.infer.bbox
        assert x1 < x2
        assert y1 < y2

        infer_id = self.infer.inference_id

        dx = x2 - x1
        dy = y2 - y1
        bbox_area_px = dx * dy
        if self.verbose:
            print(f"  inference_id {infer_id}: bbox {self.infer.bbox}")
            print(f"  inference_id {infer_id}: bbox_area_px {bbox_area_px}")
            print(f"  inference_id {infer_id}: min_area_px  {self.min_area_px}")
        if bbox_area_px < self.min_area_px:
            return False

        kps_shape = self.infer.kps.shape
        if self.verbose:
            print(f"  inference_id {infer_id}: kps {self.infer.kps}")
            print(f"  inference_id {infer_id}: kps shape {kps_shape}")
        if kps_shape != (5, 2):
            return False

        return True
