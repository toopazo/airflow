"""
Script para representar ls inferencia de una cara
"""

# import os
# from pathlib import Path
import json
import numpy as np
from copy import deepcopy


class Inference:
    """
    Clase que corresponde a una fila de la tabla inference de la base de datos
    """

    def __init__(self):
        self.inference_id = 0
        self.frame_id = 0
        self.bbox = 0
        self.kps = 0
        self.det_score = 0
        self.landmark_3d_68 = 0
        self.pose = 0
        self.landmark_2d_106 = 0
        self.gender = 0
        self.age = 0
        self.embedding = 0

    def parse_row(self, row: list):
        if len(row) == 11:
            for i, e in enumerate(row):
                # i_str = str(i).rjust(2)
                if isinstance(e, int) or isinstance(e, float):
                    # e_len = "1".ljust(10)
                    # e_type = str(type(e)).ljust(25)
                    # e_shape = "(1,)".ljust(10)
                    pass
                else:
                    # e_len = str(len(e)).ljust(10)
                    e = np.array(json.loads(e))
                    # e_type = str(type(e)).ljust(25)
                    # e_shape = str(e.shape).ljust(10)
                # print(f"  row[{i_str}] type {e_type} shape {e_shape}")

                if i == 0:
                    self.inference_id = deepcopy(e)
                elif i == 1:
                    self.frame_id = deepcopy(e)
                elif i == 2:
                    self.bbox = deepcopy(e)
                elif i == 3:
                    self.kps = deepcopy(e)
                elif i == 4:
                    self.det_score = deepcopy(e)
                elif i == 5:
                    self.landmark_3d_68 = deepcopy(e)
                elif i == 6:
                    self.pose = deepcopy(e)
                elif i == 7:
                    self.landmark_2d_106 = deepcopy(e)
                elif i == 8:
                    self.gender = deepcopy(e)
                elif i == 9:
                    self.age = deepcopy(e)
                elif i == 10:
                    self.embedding = deepcopy(e)
                else:
                    raise IndexError
        else:
            raise IndexError

    def print_info(self):
        print(f"  inference_id    value {self.inference_id}")
        print(f"  frame_id        value {self.frame_id}")
        print(f"  bbox            shape {self.bbox.shape}")
        print(f"  kps             shape {self.kps.shape}")
        print(f"  det_score       value {self.det_score}")
        print(f"  landmark_3d_68  shape {self.landmark_3d_68.shape}")
        print(f"  pose            shape {self.pose.shape}")
        print(f"  landmark_2d_106 shape {self.landmark_2d_106.shape}")
        print(f"  gender          value {self.gender}")
        print(f"  age             value {self.age}")
        print(f"  embedding       shape {self.embedding.shape}")
