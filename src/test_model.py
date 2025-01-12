from glob import glob
from PIL import Image as PilImage
from matplotlib.pyplot import imshow, show
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor
from numpy import ndarray
from torch.nn.functional import interpolate

from utils import ID2LABEL, LABEL2ID, IMAGE_SIZE


class DeepFakeDetector:
    def __init__(self, model_path:str) -> None:
        self.image_segmentation_model = (
            SegformerForSemanticSegmentation.from_pretrained(
                model_path,
                id2label=ID2LABEL,
                label2id=LABEL2ID,
            )
        )
        self.processor = SegformerImageProcessor(
            do_resize=False,
        )

    def predict_masks(self, images: list[ndarray]) -> ndarray:
        inputs = self.processor(images=images, return_tensors="pt")
        outputs = self.image_segmentation_model(**inputs)
        upsampled_logits = interpolate(
           outputs.logits,
           size=IMAGE_SIZE,
           mode="bilinear",
           align_corners=False,
        ).argmax(dim=1)
        masks = [logits.numpy() for logits in upsampled_logits]                
        return [mask.astype("uint8") for mask in masks]



detector = DeepFakeDetector(model_path="models/saved/mit-b0_deepfake_detector")

images = [PilImage.open(path).convert("RGB") for path in glob("data/images/test/*.png")]
masks = detector.predict_masks(images=images)


for image,mask in zip(images,masks):
    imshow(image)
    imshow(mask, alpha=0.5)
    show()
