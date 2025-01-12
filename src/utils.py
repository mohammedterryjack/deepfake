from glob import glob
from os import rename
from pathlib import Path

from pandas import DataFrame
from transformers import (
    SegformerImageProcessor,
    Trainer,
)
from diffusers import DiffusionPipeline
from datasets import Dataset, DatasetDict, Image, load_dataset
from evaluate import load as load_metric
from torch import from_numpy, no_grad
from torch.nn.functional import interpolate
from numpy import ones


BASE_MODEL = "nvidia/mit-b0"
IOU_METRIC = load_metric("mean_iou")
PROCESSOR = SegformerImageProcessor(
    do_resize=False,
)
ID2LABEL = {0:"Unlabelled",1:"Real",2:"Fake"}
LABEL2ID = {label:i for i,label in ID2LABEL.items()}
IMAGE_SIZE = (256,256)
REAL_DATASETS = (
    "bitmind/dtd",  #texture patterns of objects
    "bitmind/open-image-v7-256", #random images from internet, poor quality
    "bitmind/celeb-a-hq", #celebrity faces
    "bitmind/ffhq-256", #random people faces
    "bitmind/bm-real", #random photos of people, landscapes, etc
    "bitmind/MS-COCO-unique-256", #random people and objects
    "bitmind/AFHQ", #cat faces
    "bitmind/lfw", #celebrity faces
    "bitmind/caltech-256",  #furniture 
    "bitmind/caltech-101", #animals
)
REAL_DATASETS_WITH_CAPTIONS = (
    "jackyhate/text-to-image-2M", #real images
    "lambdalabs/naruto-blip-captions" #anime images
)
IMAGE_GENERATOR_MODELS = (
    "stabilityai/stable-diffusion-xl-base-1.0",  #realistic
    "SG161222/RealVisXL_V4.0", #photo realistic 
    "Corcelio/mobius", #realistic movie scenes
    "black-forest-labs/FLUX.1-dev", #realistic objects
    "prompthero/openjourney-v4", #3D fantasy style
    "cagliostrolab/animagine-xl-3.1" #Anime style 
)
REAL_FAKE_DATASET = "dragonintelligence/CIFAKE-image-dataset"

def train_with_error_handling(
    trainer: Trainer,
    logs_path: str,
    checkpoint_path: str,
    model_save_path: str,
    resume_from_checkpoint: bool,
) -> None:
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except ValueError as e:
        print(e)
        if "No valid checkpoint found" in str(e):
            return train_with_error_handling(
                trainer=trainer,
                logs_path=logs_path,
                checkpoint_path=checkpoint_path,
                model_save_path=model_save_path,
                resume_from_checkpoint=False,
            )
    except PermissionError as e:
        print(e)
        for directoryname in glob(f"{checkpoint_path}/tmp-*"):
            rename(directoryname, directoryname.replace("tmp-checkpoint", "checkpoint"))
        return train_with_error_handling(
            trainer=trainer,
            logs_path=logs_path,
            checkpoint_path=checkpoint_path,
            model_save_path=model_save_path,
            resume_from_checkpoint=True,
        )
    except KeyboardInterrupt:
        print("Saving...")
    except Exception as e:
        print(e)
        print("Saving...")
    trainer.save_model(model_save_path)
    logs = DataFrame(trainer.state.log_history)
    logs.to_csv(f"{logs_path}.csv")



def image_segmentation_metric(eval_pred):
    with no_grad():
        logits, expected = eval_pred
        upsampled_logits = interpolate(
            from_numpy(logits), size=IMAGE_SIZE, mode="bilinear", align_corners=False
        ).argmax(dim=1)
        predicted = upsampled_logits.detach().cpu().numpy()
        metrics = IOU_METRIC._compute(
            predictions=predicted,
            references=expected,
            num_labels=2,
            ignore_index=0,
            reduce_labels=False,
        )

        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        metrics.update(
            {f"accuracy_category{i}": v for i, v in enumerate(per_category_accuracy)}
        )
        metrics.update({f"iou_category{i}": v for i, v in enumerate(per_category_iou)})
        return metrics

def transform_image_and_mask(sample: dict):
    images = [x for x in sample["pixel_values"]]
    masks = [x for x in sample["label"]]
    return PROCESSOR(images, masks)


def convert_to_huggingface_dataset(
    path_to_images: str,
    test_split: float = 0.02,
) -> DatasetDict:
    real_tiles = sorted(glob(f"{path_to_images}/real/*.png"))
    fake_tiles = sorted(glob(f"{path_to_images}/fake/*.png"))
    real_masks = [ones(IMAGE_SIZE)*LABEL2ID['Real'] for _ in range(len(real_tiles))]
    fake_masks = [ones(IMAGE_SIZE)*LABEL2ID['Fake'] for _ in range(len(fake_tiles))]

    dataset = Dataset.from_dict(
        {
            "label": real_masks + fake_masks,
            "pixel_values": real_tiles + fake_tiles,
        }
    )
    dataset = dataset.cast_column("pixel_values", Image())
    dataset = dataset.cast_column("label", Image())
    dataset_dict = DatasetDict({"dataset": dataset})
    dataset_dict = dataset_dict.shuffle(seed=1)
    dataset_train_test = dataset_dict["dataset"].train_test_split(
        test_size=test_split
    )
    dataset_train_test.set_transform(transform_image_and_mask)
    return dataset_train_test


def load_huggingface_dataset(
    dataset_name:str,
    path_to_images: str,
    test_split: float = 0.02,
) -> DatasetDict:
    data = load_dataset(dataset_name)

    images = [image.resize(IMAGE_SIZE) for image in data['train']['image'][:10]]
    masks = [ones(IMAGE_SIZE)*LABEL2ID['Real'] for _ in range(len(images))]

    dataset = Dataset.from_dict(
        {
            "label": masks,
            "pixel_values": images
        }
    )
    dataset = dataset.cast_column("pixel_values", Image())
    dataset = dataset.cast_column("label", Image())
    dataset_dict = DatasetDict({"dataset": dataset})
    dataset_dict = dataset_dict.shuffle(seed=1)
    dataset_train_test = dataset_dict["dataset"].train_test_split(
        test_size=test_split
    )
    dataset_train_test.set_transform(transform_image_and_mask)
    return dataset_train_test


def load_cifake_dataset() -> DatasetDict:
    #NOTE: too small, images are 30x30 pixels!!!

    data = load_dataset(REAL_FAKE_DATASET)
    
    images = [image.resize(IMAGE_SIZE) for image in data['train']['image']]
    masks = [ones(IMAGE_SIZE)*(LABEL2ID['Fake'],LABEL2ID['Real'])[label] for label in data['train']['label']]

    dataset = Dataset.from_dict(
        {
            "label": masks,
            "pixel_values": images
        }
    )
    dataset = dataset.cast_column("pixel_values", Image())
    dataset = dataset.cast_column("label", Image())
    dataset_dict = DatasetDict({"dataset": dataset})
    dataset_dict = dataset_dict.shuffle(seed=1)
    dataset_train_test = dataset_dict["dataset"].train_test_split(
        test_size=test_split
    )
    dataset_train_test.set_transform(transform_image_and_mask)
    return dataset_train_test

def load_true_fake_anime_dataset(
    image_generation_model:str,
    n:int,
    test_split: float = 0.02,
    generate:bool=False
) -> DatasetDict:
    data = load_dataset(REAL_DATASETS_WITH_CAPTIONS[1])
    real_images = [image.resize(IMAGE_SIZE) for image in data['train']['image'][:n]]
    captions = data['train']['text'][:n]
    if generate:
        for caption,real_image in zip(captions,real_images):
            filename = Path(f"data/images/real/{caption}.png")
            if filename.exists():
                continue
            print(filename)
            real_image.save(filename)        

        image_generator = DiffusionPipeline.from_pretrained(image_generation_model)
        for caption in captions:
            filename = Path(f"data/images/fake/{image_generation_model}/{caption}.png")
            if filename.exists():
                continue
            print(filename)
            filename.parent.mkdir(parents=True, exist_ok=True)
            image = image_generator(caption).images[0]
            image256=image.resize(IMAGE_SIZE)
            image256.save(filename)

    fake_images = list(glob(f"data/images/fake/{image_generation_model}/*.png"))

    real_masks = [ones(IMAGE_SIZE)*LABEL2ID['Real'] for _ in range(len(real_images))]
    fake_masks = [ones(IMAGE_SIZE)*LABEL2ID['Fake'] for _ in range(len(fake_images))]

    dataset = Dataset.from_dict(
        {
            "label": real_masks + fake_masks,
            "pixel_values": real_images + fake_images
        }
    )
    dataset = dataset.cast_column("pixel_values", Image())
    dataset = dataset.cast_column("label", Image())
    dataset_dict = DatasetDict({"dataset": dataset})
    dataset_dict = dataset_dict.shuffle(seed=1)
    dataset_train_test = dataset_dict["dataset"].train_test_split(
        test_size=test_split
    )
    dataset_train_test.set_transform(transform_image_and_mask)
    return dataset_train_test
