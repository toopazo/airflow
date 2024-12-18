from PIL import Image, ImageShow
import cv2
import numpy as np

# import insightface
from insightface.app import FaceAnalysis

# from insightface.data import get_image as ins_get_image


class ProcessFrame:
    def __init__(self):
        model_pack_name = "buffalo_l"
        self.app = FaceAnalysis(
            name=model_pack_name,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        self.app.prepare(ctx_id=0, det_thresh=0.5, det_size=(640, 640))

        self.key_list = [
            "bbox",
            "kps",
            "det_score",
            "landmark_3d_68",
            "pose",
            "landmark_2d_106",
            "gender",
            "age",
            "embedding",
        ]
        self.key_bbox = "bbox"
        self.key_kps = "kps"
        self.key_det_score = "det_score"
        self.key_landmark_3d_68 = "landmark_3d_68"
        self.key_pose = "pose"
        self.key_landmark_2d_106 = "landmark_2d_106"
        self.key_gender = "gender"
        self.key_age = "age"
        self.key_embedding = "embedding"

    def inference_pil(self, image: Image.Image):
        img = self.pil_to_cv2(image)
        faces = self.app.get(img)
        # faces[0] keys dict_keys(['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'])
        if len(faces) > 0:
            print(f"faces {type(faces)}")
            print(f"faces[0] type {type(faces[0])}")
            print(f"faces[0] keys {faces[0].keys()}")
        return faces

    def inference_cv2(self, img: np.ndarray):
        faces = self.app.get(img)
        # faces[0] keys dict_keys(['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'])
        # print(f"faces {type(faces)}")
        # print(f"faces[0] type {type(faces[0])}")
        # print(f"faces[0] keys {faces[0].keys()}")
        return faces

    def draw_faces_pil(self, image: Image.Image, faces: list):
        img = self.pil_to_cv2(image)
        imgd = self.app.draw_on(img, faces)
        return self.cv2_to_pil(imgd)

    def draw_faces_cv2(self, img: np.ndarray, faces: list):
        imgd = self.app.draw_on(img, faces)
        return imgd

    def pil_to_cv2(self, image: Image.Image):
        img = np.array(image)
        img = img[:, :, ::-1].copy()
        return img

    def cv2_to_pil(self, img: np.ndarray):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(img.copy())
        return image
