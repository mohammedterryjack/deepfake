from transformers import AutoProcessor, AutoModelForVisualQuestionAnswering
from PIL import Image
from torch import float16

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-6.7b-coco")
model = AutoModelForVisualQuestionAnswering.from_pretrained("Salesforce/blip2-opt-6.7b-coco")

image = Image.open("data/images/fake/stable_diffusion_xl_astronaut.png")
inputs = processor(
    images=image, 
    return_tensors="pt"
).to("cpu", float16)
generated_ids = model.generate(**inputs)
generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

print(generated_text)
