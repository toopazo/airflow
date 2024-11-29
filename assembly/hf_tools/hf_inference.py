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
from transformers import pipeline


class HfInference:
    """
    Clase de demo para HuggingFace

    https://huggingface.co/docs/transformers/en/pipeline_tutorial

    """

    def __init__(self) -> None:

        print()
        print("Audio to text")
        pipe = pipeline(model="openai/whisper-large-v2")
        # pipe = pipeline(task="automatic-speech-recognition")
        result = pipe(
            "https://huggingface.co/datasets/Narsil/asr_dummy/resolve/main/mlk.flac"
        )
        print(f"result {result}")

        print()
        print("Text: run using iterator")
        pipe = pipeline(model="openai-community/gpt2")
        generated_characters = 0
        for out in pipe(self.data_iterator()):
            generated_characters += len(out[0]["generated_text"])

        print()
        print("Image classification")
        pipe = pipeline(model="google/vit-base-patch16-224")
        result = pipe(
            images="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"
        )
        print(f"result {result}")

        print()
        print("VQA: visual question answering")
        pipe = pipeline(model="impira/layoutlm-document-qa")
        result = pipe(
            image="https://huggingface.co/spaces/impira/docquery/resolve/2359223c1837a7587402bda0f2643382a6eefeab/invoice.png",
            question="What is the invoice number?",
        )
        print(f"result {result}")

    def data_iterator(self):
        for i in range(10):
            yield f"My example {i}"


if __name__ == "__main__":
    HfInference()
