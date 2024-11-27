from pathlib import Path
import numpy as np

from PIL import Image, ImageShow
from datasets import load_dataset, get_dataset_split_names

from transformers import AutoFeatureExtractor

# from datasets import load_dataset, Image


import torch
from torchvision.transforms import Compose, ColorJitter, ToTensor, RandomRotation
from torch.utils.data import DataLoader


class HFDemo:
    """Clase de demo para HuggingFace"""

    def __init__(self) -> None:
        splits = get_dataset_split_names("beans")
        print(f"splits {splits}")

        dataset = load_dataset("beans", split="train")

        print(dataset)
        print(dataset[0])

        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        dataset = load_dataset("beans", split="train")

        image_0 = dataset[0]["image"]
        ImageShow.show(image_0, title="Original image 0")
        # feature_0 = feature_extractor(image_0)
        # print(np.array(feature_0["pixel_values"]).shape)

        dataset.set_transform(self.transforms_rotate)
        image_0 = dataset[0]["pixel_values"]
        ImageShow.show(image_0, title="Rotated image 0")

        # # dataset = dataset.cast_column("image", Image(mode="RGB"))
        # dataset = dataset.with_transform(self.transforms)
        # self.dataset = dataset

        # dataloader = DataLoader(dataset, collate_fn=self.collate_fn, batch_size=4)
        # self.dataloader = dataloader

    def transforms_rotate(self, examples):
        rotate = RandomRotation(degrees=(0, 90))
        examples["pixel_values"] = [rotate(image) for image in examples["image"]]
        return examples

    def transforms(self, examples):
        jitter = Compose([ColorJitter(brightness=0.5, hue=0.5), ToTensor()])
        examples["pixel_values"] = [
            jitter(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    def collate_fn(self, examples):
        images = []
        labels = []
        for example in examples:
            images.append((example["pixel_values"]))
            labels.append(example["labels"])

        pixel_values = torch.stack(images)
        labels = torch.tensor(labels)
        return {"pixel_values": pixel_values, "labels": labels}

    def process_image(self, image):
        pass


if __name__ == "__main__":
    HFDemo()
