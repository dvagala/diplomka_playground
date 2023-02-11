# TF2 version
import io
import numpy as np
import requests
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import cv2


IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512 

# Download the model from Tensorflow Hub.
keras_layer = hub.KerasLayer('https://tfhub.dev/google/edgetpu/vision/autoseg-edgetpu/default_argmax/s/1')
model = tf.keras.Sequential([keras_layer])
model.build([None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])

# Preprocess image.
# image_url = "https://storage.googleapis.com/tf_model_garden/models/edgetpu/images/ADE_train_00016869.jpeg"
# image_response = requests.get(image_url)
# image = PILImage.open(io.BytesIO(image_response.content)).convert('RGB')
image = PILImage.open('photo.jpg').convert('RGB')

min_dim = min(image.size[0], image.size[1])
image = image.resize((IMAGE_WIDTH * image.size[0] // min_dim,
                      IMAGE_HEIGHT * image.size[1] // min_dim))
input_data = np.expand_dims(image, axis=0)
input_data = input_data[:, :IMAGE_WIDTH,:IMAGE_HEIGHT, :]
input_data = input_data.astype(np.float) / 128 - 0.5

# Run segmentation.
output_data = model(input_data)

# cv2.imshow("img", np.squeeze(output_data))

plt.imshow(output_data.numpy()[0])
# plt.imshow(np.array(output_data, dtype=np.uint8))

plt.show()


