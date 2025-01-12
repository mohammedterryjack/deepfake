from utils import REAL_DATASETS
from datasets import load_dataset
from numpy import array

# from utils import convert_to_huggingface_dataset

# dataset = convert_to_huggingface_dataset(
#     path_to_images="data/images",
# )
# image = dataset["train"]['pixel_values'][0]
# image.show()
# image256 = image.resize((256,256))
# image256.show()


# for name in REAL_DATASETS:
#     print(name)
#     dataset = load_dataset(name)
#     image = dataset['train']['image'][0]
#     image256 = image.resize((256,256))
#     #image256.save("data/real", "PNG")
#     image256.show()
#     #print(array(image256).shape)
#     break

