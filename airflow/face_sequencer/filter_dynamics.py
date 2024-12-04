"""
Script para filtrar caras por dinamica
"""

# import os
# from pathlib import Path
from copy import deepcopy
import numpy as np
from airflow.database.table_inference import Inference
from airflow.utils.bbox_geometry import BboxGeometry


class DynamicFilter:
    def __init__(self):
        self.verbose = False

        self.sequence: list[int] = []
        self.previous_bbox = np.array([])

        self.max_displacement_px = 100
        self.max_iou_overlap = 0.1
        self.bbox_geometry = BboxGeometry()

    def initialize(self, infer: Inference):
        self.sequence = [infer.inference_id]
        self.previous_bbox = deepcopy(infer.bbox)

    def apply(self, infer_list: list[Inference]):
        x1, y1, x2, y2 = self.previous_bbox
        assert x1 < x2
        assert y1 < y2

        dx = x2 - x1
        dy = y2 - y1
        xc = int(x1 + dx / 2)
        yc = int(y1 + dy / 2)
        previous_center = np.array([xc, yc])

        # Chose candidate based on min displacement
        dist_list = []
        for infer in infer_list:
            infer_id = infer.inference_id
            x1, y1, x2, y2 = infer.bbox
            dx = x2 - x1
            dy = y2 - y1
            xc = int(x1 + dx / 2)
            yc = int(y1 + dy / 2)
            infer_center = np.array([xc, yc])

            distance = np.linalg.norm(infer_center - previous_center)
            dist_list.append(distance)

        arg_min_dist = np.argmin(dist_list).squeeze()
        if self.verbose:
            print(f"arg_min_dist {arg_min_dist} in dist_list {dist_list}")
        displacement = dist_list[arg_min_dist]
        current_infer = deepcopy(infer_list[arg_min_dist])
        current_iou = self.bbox_geometry.get_iou(current_infer.bbox, self.previous_bbox)
        current_infer_id = current_infer.inference_id
        current_bbox = current_infer.bbox

        if self.verbose:
            print(f"  inference_id {current_infer_id}: iou {current_iou}")
            print(f"  inference_id {current_infer_id}: displacement {displacement}")
            print(
                f"  inference_id {current_infer_id}: max_displacement_px {self.max_displacement_px}"
            )
        if displacement > self.max_displacement_px:
            return False

        # Check iou for occlusion and/or confusion risks with the rest of candidates
        infer_list.pop(arg_min_dist)
        for infer in infer_list:
            infer_id = infer.inference_id
            iou = self.bbox_geometry.get_iou(infer.bbox, self.previous_bbox)

            if self.verbose:
                print(f"  inference_id {infer_id}: iou {iou}")
                print(
                    f"  inference_id {infer_id}: max_iou_overlap {self.max_iou_overlap}"
                )
            if iou > self.max_iou_overlap:
                return False

        self.sequence.append(current_infer_id)
        self.previous_bbox = current_bbox

        return True
