from diffusers import DiffusionPipeline

from utils import IMAGE_GENERATOR_MODELS

pipe = DiffusionPipeline.from_pretrained(IMAGE_GENERATOR_MODELS[-1])

prompt = "A falcon flying above a galloping horse"
image = pipe(prompt).images[0]
image.show()
