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

        self.sequence = {"frame_id": [], "inference_id": []}
        self.sequence_key_frame_id = "frame_id"
        self.sequence_key_inference_id = "inference_id"
        self.sequence_key_bbox = "bbox"

        self.previous_bbox = np.array([])

        self.max_displacement_px = 100
        self.max_iou_overlap = 0.1
        self.bbox_geometry = BboxGeometry()

    def initialize(self, infer: Inference):
        self.sequence = {
            self.sequence_key_frame_id: [infer.frame_id],
            self.sequence_key_inference_id: [infer.inference_id],
            self.sequence_key_bbox: [infer.bbox],
        }
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

        if self.verbose:
            cur_infid = current_infer.inference_id
            current_iou = self.bbox_geometry.get_iou(
                current_infer.bbox, self.previous_bbox
            )
            print(f"  inference_id {cur_infid}: iou {current_iou}")
            print(f"  inference_id {cur_infid}: displacement {displacement}")
            print(
                f"  inference_id {cur_infid}: max_displacement_px {self.max_displacement_px}"
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

        self.sequence[self.sequence_key_frame_id].append(current_infer.frame_id)
        self.sequence[self.sequence_key_inference_id].append(current_infer.inference_id)
        self.sequence[self.sequence_key_bbox].append(current_infer.bbox)
        self.previous_bbox = current_infer.bbox

        return True


class SequenceTracker:
    def __init__(self, infer_list: list[Inference]):
        self.active_seq_list: list[DynamicFilter] = []
        for infer in infer_list:
            dynfil = DynamicFilter()
            dynfil.initialize(infer=infer)
            self.active_seq_list.append(dynfil)

    def update(self, infer_list: list[Inference]) -> tuple:
        left_overs = [e.inference_id for e in infer_list]
        terminated_seq_ix = []
        terminated_seq_list = []
        for ix, dynfil in enumerate(self.active_seq_list):
            approved = dynfil.apply(infer_list=infer_list)

            seq_inference_id = dynfil.sequence[dynfil.sequence_key_inference_id]
            if approved:
                seq_len = f"{len(seq_inference_id)}".rjust(6)
                print(
                    f"  Sequence {ix} with len {seq_len} was accepted (continued)  by dynamics"
                )
                left_overs.remove(seq_inference_id[-1])
            else:
                seq_len = f"{len(seq_inference_id)}".rjust(6)
                print(
                    f"  Sequence {ix} with len {seq_len} was rejected (terminated) by dynamics"
                )
                terminated_seq_ix.append(ix)
                terminated_seq_list.append(deepcopy(dynfil))
                # raise RuntimeError

        # Handle terminated sequences
        if len(terminated_seq_ix) > 0:
            print(f"  ----> Handling terminated sequences {terminated_seq_ix}")
            for rm_ix in terminated_seq_ix:
                rm_infer = self.active_seq_list.pop(rm_ix)
                rm_seq_inference_id = rm_infer.sequence[rm_infer.sequence_key_frame_id]
                print(
                    f"  ----> Removing sequence terminated at inference_id {rm_seq_inference_id[-1]}"
                )

        # Handle left overs
        if len(left_overs) > 0:
            # raise RuntimeError
            print(f"  ----> Handling left_overs {left_overs}")
            for left_over_id in left_overs:
                for infer in infer_list:
                    if infer.inference_id == left_over_id:
                        print(
                            f"  ----> A new sequence was started from inference_id {left_over_id}"
                        )
                        dynfil = DynamicFilter()
                        dynfil.initialize(infer=infer)
                        self.active_seq_list.append(dynfil)

        return self.active_seq_list, terminated_seq_list