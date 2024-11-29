from pathlib import Path
import numpy as np
from PIL import Image, ImageShow

from datasets import load_dataset, get_dataset_split_names

import albumentations
from torchvision.transforms import (
    Compose,
    ColorJitter,
    ToTensor,
    RandomRotation,
    RandomResizedCrop,
    Normalize,
)

# HuggingFace
from transformers import AutoFeatureExtractor
from transformers import AutoModelForImageClassification
from transformers import Trainer
from transformers import TrainingArguments
from transformers import AutoImageProcessor
from transformers import DefaultDataCollator
import evaluate

# from torch.utils.data import DataLoader


class HfTrainVit:
    """
    Clase de demo para HuggingFace

    https://huggingface.co/docs/transformers/main/en/main_classes/image_processor
    https://huggingface.co/docs/transformers/en/model_doc/vit

    https://huggingface.co/docs/datasets/en/about_dataset_features
    https://huggingface.co/docs/datasets/en/image_classification

    https://huggingface.co/docs/transformers/en/tasks/image_classification

    """

    def __init__(self) -> None:

        # features = self.calculate_features(dataset1)
        # print(features.shape)

        print()
        print("Custom dataset")
        custom_dataset = self.custom_dataset()
        print(custom_dataset)
        data0 = custom_dataset["test"][0]
        image0 = data0["image"]
        label0 = data0["label"]
        print(f"label0 {label0}")
        ImageShow.show(image0)

        # custom_dataset = custom_dataset.map(
        #     self.transform_resize, remove_columns=["image"], batched=True
        # )
        # print(custom_dataset)
        # data0 = custom_dataset["test"][0]
        # image0 = data0["pixel_values"]
        # label0 = data0["label"]
        # print(label0)
        # ImageShow.show(image0)

        print("transform_augmentation")
        # custom_dataset.set_transform(self.transform_augmentation)
        dataset_ta = custom_dataset.with_transform(self.transform_augmentation)
        # custom_dataset = custom_dataset.map(
        #     self.transform_augmentation, batched=True
        # ).remove_columns(["image"])
        print(dataset_ta)
        data0 = dataset_ta["test"][0]
        image0 = data0["pixel_values"]
        print(f"image0 {image0} type {type(image0)}")
        image0 = Image.fromarray(image0)
        label0 = data0["label"]
        print(f"label0 {label0}")
        ImageShow.show(image0)

        print("transform_to_tensor")
        # custom_dataset.set_transform(self.transform_to_tensor)
        # custom_dataset["image"] = custom_dataset["pixel_values"]
        # custom_dataset = custom_dataset.map(
        #     self.transform_to_tensor, batched=True
        # ).remove_columns(["image"])
        dataset_tp = custom_dataset.with_transform(self.transform_preprocess)
        print(dataset_tp)
        data0 = dataset_tp["test"][0]
        image0 = data0["pixel_values"]
        label0 = data0["label"]
        print(f"label0 {label0}")
        print(f"image0 {image0}")

        print("custom_dataset")
        labels = dataset_tp["train"].features["label"].names
        label2id, id2label = dict(), dict()
        for i, label in enumerate(labels):
            label2id[label] = str(i)
            id2label[str(i)] = label
        print(f"labels {labels}")
        print(dataset_tp)

        checkpoint = "google/vit-base-patch16-224-in21k"
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)

        data_collator = DefaultDataCollator()

        model = AutoModelForImageClassification.from_pretrained(
            checkpoint,
            num_labels=len(labels),
            id2label=id2label,
            label2id=label2id,
        )

        training_args = TrainingArguments(
            output_dir="my_awesome_food_model",
            remove_unused_columns=False,
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=5e-5,
            per_device_train_batch_size=16,
            gradient_accumulation_steps=4,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            warmup_ratio=0.1,
            logging_steps=10,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            push_to_hub=False,
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=dataset_tp["train"],
            eval_dataset=dataset_tp["test"],
            processing_class=image_processor,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()

        # custom_dataset.save_to_disk("clasif_desal.hf")

        # # dataset = dataset.cast_column("image", Image(mode="RGB"))
        # dataset = dataset.with_transform(self.transforms)
        # self.dataset = dataset

        # dataloader = DataLoader(dataset, collate_fn=self.collate_fn, batch_size=4)
        # self.dataloader = dataloader

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = evaluate.load("accuracy")
        return accuracy.compute(predictions=predictions, references=labels)

    def test_beans(self):
        splits = get_dataset_split_names("beans")
        print(f"splits {splits}")

        beans_dataset = load_dataset("beans", split="train")

        print(beans_dataset)
        print(beans_dataset[0])

        beans_dataset = load_dataset("beans", split="train")

        image_0 = beans_dataset[0]["image"]
        ImageShow.show(image_0, title="Original image 0")

        beans_dataset.set_transform(self.transform_rotate)
        image_0 = beans_dataset[0]["pixel_values"]
        ImageShow.show(image_0, title="Rotated image 0")

    def custom_dataset(self):
        data_path = Path("/home/toopazo/repos_s3/clasif_desal/")
        dataset = load_dataset("imagefolder", data_dir=data_path)
        return dataset

    def calculate_features(self, dataset):
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            "google/vit-base-patch16-224-in21k"
        )
        # image_0 = dataset[0]["image"]
        # feature_0 = feature_extractor(image_0)
        features = [feature_extractor(data["image"]) for data in dataset]
        return np.array(features).squeeze()

    def transform_to_tensor(self, examples):
        totensor = Compose([ToTensor()])
        examples["pixel_values"] = [
            totensor(img.convert("RGB")) for img in examples["image"]
        ]
        # del examples["image"]
        return examples

    def transform_rotate(self, examples):
        rotate = RandomRotation(degrees=(0, 90))
        examples["pixel_values"] = [rotate(image) for image in examples["image"]]
        return examples

    def transform_color_jitter(self, examples):
        jitter = Compose([ColorJitter(brightness=0.5, hue=0.5), ToTensor()])
        examples["pixel_values"] = [
            jitter(image.convert("RGB")) for image in examples["image"]
        ]
        return examples

    def transform_resize(self, examples):
        examples["pixel_values"] = [
            image.convert("RGB").resize((250, 250)) for image in examples["image"]
        ]

        return examples

    def transform_augmentation(self, examples):
        transform = albumentations.Compose(
            [
                albumentations.Resize(width=255, height=255),
                albumentations.RandomCrop(width=250, height=250),
                albumentations.HorizontalFlip(p=0.5),
                albumentations.RandomBrightnessContrast(p=0.2),
                # albumentations.Resize(width=255, height=255),
            ]
        )
        examples["pixel_values"] = [
            transform(image=np.array(image))["image"] for image in examples["image"]
        ]

        return examples

    def transform_preprocess(self, examples):
        # transform = albumentations.Compose(
        #     [
        #         albumentations.Resize(width=255, height=255),
        #         albumentations.RandomCrop(width=250, height=250),
        #         albumentations.HorizontalFlip(p=0.5),
        #         albumentations.RandomBrightnessContrast(p=0.2),
        #         # albumentations.Resize(width=255, height=255),
        #     ]
        # )
        # examples["pixel_values"] = [
        #     transform(image=np.array(image))["image"] for image in examples["image"]
        # ]

        checkpoint = "google/vit-base-patch16-224-in21k"
        image_processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)

        normalize = Normalize(
            mean=image_processor.image_mean, std=image_processor.image_std
        )

        size = (
            image_processor.size["shortest_edge"]
            if "shortest_edge" in image_processor.size
            else (image_processor.size["height"], image_processor.size["width"])
        )

        _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

        examples["pixel_values"] = [
            _transforms(img.convert("RGB")) for img in examples["image"]
        ]
        del examples["image"]
        return examples

    # def collate_fn(self, examples):
    #     images = []
    #     labels = []
    #     for example in examples:
    #         images.append((example["pixel_values"]))
    #         labels.append(example["labels"])

    #     pixel_values = torch.stack(images)
    #     labels = torch.tensor(labels)
    #     return {"pixel_values": pixel_values, "labels": labels}

    # def process_image(self, image):
    #     pass


if __name__ == "__main__":
    HfTrainVit()
