from PIL import Image, ImageShow
import cv2
import numpy as np

# import insightface
from insightface.app import FaceAnalysis

# from insightface.data import get_image as ins_get_image

model_pack_name = "buffalo_l"
app = FaceAnalysis(
    name=model_pack_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)
app.prepare(ctx_id=0, det_size=(640, 640))

# img = ins_get_image("t1")
image = Image.open("t1.jpg")
img = np.array(image)
img = img[:, :, ::-1].copy()

faces = app.get(img)
# faces[0] keys dict_keys(['bbox', 'kps', 'det_score', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'gender', 'age', 'embedding'])
print(f"faces {type(faces)}")
print(f"faces[0] type {type(faces[0])}")
print(f"faces[0] keys {faces[0].keys()}")

imgd = app.draw_on(img, faces)
# cv2.imwrite("./t1_output.jpg", imgd)
imgd = cv2.cvtColor(imgd, cv2.COLOR_BGR2RGB)
imaged = Image.fromarray(imgd)
ImageShow.show(imaged)
