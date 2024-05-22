from transformers import pipeline
from PIL import Image
import requests

# load pipe
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-large-hf")

# load image
url = '/DATA2/mayur/pawscan/code/babu/datasets/CASIA-B/CASIA-B/GaitDatasetB-silh/001/bg-01/000'
image = Image.open(requests.get(url, stream=True).raw)

# inference
depth = pipe(image)["depth"]
print('url of image :',url)
print("Data type of image:", type(image))
print("Data type of depth:", type(depth))