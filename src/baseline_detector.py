"""Baseline DeepFake detector for comparison - stores all real image hashes for lookups"""

from datasets import load_dataset
from imagehash import dhash
from PIL import Image, PngImagePlugin
from utils import REAL_DATASETS
from pathlib import Path
from enum import Enum


class DeepFake(Enum):
    REAL=1
    FAKE=0

def classify(image:Image) -> DeepFake:
    """0=Fake, 1=Real"""
    encoded_image = dhash(image.resize((256,256)))
    for dataset_name in REAL_DATASETS:
        load_path = Path(f'data/images/real/{Path(dataset_name).stem}.txt')
        with load_path.open() as f:
            encoded_images = f.readlines()
        if encoded_image in encoded_images:
            print(dataset_name)
            return DeepFake.REAL
    return DeepFake.FAKE


def store_image_encodings() -> None:
    PngImagePlugin.MAX_TEXT_CHUNK = 10 * 1024 * 1024
    for dataset_name in REAL_DATASETS:
        print(dataset_name)
        save_path = Path(f'data/images/real/{Path(dataset_name).stem}.txt')
        if save_path.exists():
            continue
        dataset = load_dataset(dataset_name, streaming=True)
        with save_path.open('w') as f:
            for sample in dataset['train']:
                image256 = sample['image'].resize((256,256))
                f.write(f"{dhash(image256)}\n")    

store_image_encodings()
