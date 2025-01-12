from pathlib import Path
from transformers import (
    SegformerForSemanticSegmentation,
    Trainer,
    TrainingArguments,
)

from utils import (
    BASE_MODEL, 
    ID2LABEL, 
    LABEL2ID,
    IMAGE_GENERATOR_MODELS,
    load_true_fake_anime_dataset,
    train_with_error_handling, 
    image_segmentation_metric
)

MODEL_NAME = f"{Path(BASE_MODEL).stem}_deepfake_detector"

dataset = load_true_fake_anime_dataset(
    n=10,
    image_generation_model=IMAGE_GENERATOR_MODELS[-1], 
    generate=False
)

train_with_error_handling(
    resume_from_checkpoint=True,
    logs_path=f"models/logs/{MODEL_NAME}",
    checkpoint_path=f"models/checkpoints/{MODEL_NAME}",
    model_save_path=f"models/saved/{MODEL_NAME}",
    trainer=Trainer(
        model=SegformerForSemanticSegmentation.from_pretrained(
            BASE_MODEL, id2label=ID2LABEL, label2id=LABEL2ID
        ),
        train_dataset=dataset['train'],
        eval_dataset=dataset['test'],
        compute_metrics=image_segmentation_metric,
        args=TrainingArguments(
            f"models/checkpoints/{MODEL_NAME}",
            learning_rate=0.00006,
            num_train_epochs=50,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            save_total_limit=3,
            evaluation_strategy="steps",
            save_strategy="steps",
            save_steps=20,
            eval_steps=20,
            logging_steps=1,
            eval_accumulation_steps=5,
            load_best_model_at_end=True,
            push_to_hub=False,
        ),
    ),
)