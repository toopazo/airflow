from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageShow
from airflow.utils.video_handler import VideoHandler
from airflow.database.table_inference import Inference


class ImagePainter:
    def __init__(self, video_id: int):
        self.video_handler = VideoHandler(video_id=video_id)

    # def show_frames(self, frame_i: int, frame_j: int):
    #     frames_dict = self.video_handler.get_data_frames(
    #         frame_i=frame_i, frame_j=frame_j
    #     )
    #     for frame_id, val in frames_dict.items():
    #         image = val["image"]
    #         assert isinstance(image, Image.Image)
    #         ImageShow.show(image, title=f"frame_id {frame_id}")

    # def get_frames(self, frame_i: int, frame_j: int):
    #     frames_dict = self.video_handler.get_data_frames(
    #         frame_i=frame_i, frame_j=frame_j
    #     )
    #     image_list = []
    #     for _, val in frames_dict.items():
    #         image = val["image"]
    #         assert isinstance(image, Image.Image)
    #         image_list.append(image)
    #     return image_list

    def draw_crop_list(self, bbox_list: list, frame_id_list: list, image_path: Path):
        crop_list: list[Image.Image] = []
        for bbox, frame_id in zip(bbox_list, frame_id_list):

            frames_dict = self.video_handler.get_data_frames(
                frame_i=frame_id, frame_j=frame_id + 1
            )
            image = frames_dict[frame_id]["image"]
            assert isinstance(image, Image.Image)

            left = bbox[0]
            upper = bbox[1]
            right = bbox[2]
            lower = bbox[3]
            image_crop = image.crop((left, upper, right, lower))
            crop_list.append(image_crop)

        width, height = crop_list[0].size

        img = Image.new(
            "RGB", (width * len(crop_list), height), color=(0, 0, 0)
        )  # set width, height of new image based on original image
        for i, crop in enumerate(crop_list):
            img.paste(
                crop, (i * width, 0)
            )  # the second argument here is tuple representing upper left corner
        img.save(image_path)

    def auto_draw_annotations(self, frame_i: int, frame_j: int):
        infer_dict = self.video_handler.get_data_inferences(
            frame_i=frame_i, frame_j=frame_j
        )
        frames_dict = self.video_handler.get_data_frames(
            frame_i=frame_i, frame_j=frame_j
        )

        image_list = []
        frame_ids = frames_dict.keys()
        # for k, v in frames_dict.items():
        for frame_id in frame_ids:
            k = frame_id
            v = frames_dict[k]
            image = v["image"]
            assert isinstance(image, Image.Image)
            # print(f"frames_dict[{k}][image] has size {image.size}")
            # image_list.append(image)

            bbox_list = []
            label_list = []
            v = infer_dict[k]
            infer_list = v["infer_list"]
            # print(f"infer_dict[{k}][infer_list] has len {len(infer_list)}")
            for infer in infer_list:
                assert isinstance(infer, Inference)
                # infer.print_info()

                bbox_list.append(infer.bbox)
                label_list.append("face")

            image = self.draw_annotations(
                image=image, bboxes=bbox_list, labels=label_list
            )
            ImageShow.show(image)
            image_list.append(image)

        return image_list

    def draw_on_image(
        self,
        image: Image.Image,
        bboxes: np.ndarray,
        labels: np.ndarray,
    ):
        image = self.apply_mask(image)
        image = self.draw_table_magnet(image)
        image = self.draw_annotations(image, bboxes, labels)

        return image

    def draw_table_magnet(self, image: Image.Image):

        draw = ImageDraw.Draw(image)

        table = self.table
        coords = (table[0], table[1], table[2], table[3])
        draw.rectangle(
            coords,
            outline="red",
            width=3,
        )

        magnet = self.magnet
        coords = (
            magnet[0],
            magnet[1],
            magnet[2],
            magnet[3],
        )
        draw.rectangle(
            (coords),
            outline="green",
            width=3,
        )
        # draw.text(
        #     (roi[0], roi[1]),
        #     label,
        #     font=ImageFont.truetype("font_path123"),
        # )
        return image

    def draw_annotations(self, image: Image.Image, bboxes: list, labels: list):

        # creating
        # create rectangle image
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=18)
        for bbox, label in zip(bboxes, labels):
            draw.rectangle(
                ((bbox[0], bbox[1]), (bbox[2], bbox[3])), outline="blue", width=3
            )
            draw.text(
                (bbox[0], bbox[1]),
                label,
                font=font,
            )
        return image

        # image = np.array(image)
        # image = image[:, :, ::-1].copy()
        # image = mmcv.imread(image)

        # if len(bboxes) > 0:
        #     mmcv.imshow_det_bboxes(
        #         img=image,
        #         bboxes=bboxes,
        #         labels=labels,
        #         show=False,
        #         thickness=5,
        #         font_scale=0.8,
        #         bbox_color="red",
        #         text_color="red",
        #     )

        # mmcv.imwrite(image, out_path)

    def apply_mask(self, image: Image.Image):
        # im1 = image.convert("L")
        im1 = image
        im2 = Image.fromarray(np.array(im1) * 0)

        # For cv2
        # h, w, c = im.shape

        # For PIL
        # w, h = im.size

        im3 = Image.composite(im1, im2, self.mask)
        return im3
