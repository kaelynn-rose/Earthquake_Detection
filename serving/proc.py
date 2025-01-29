


import gc
import json
import requests
import sys
import time

import numpy as np
import pandas as pd

from PIL import Image

sys.path.append('../')
sys.path.append('../../../')

import earthquake_detection.data_preprocessing as DataPreprocessing


# Load extracted raw signals
raw_signals = np.load('../../../data/STEAD/extracted_raw_signals_subsample_1000.npy')

# Load metadata
metadata = pd.read_feather('../../../data/STEAD/extracted_metadata_subsample_1000.feather')
metadata = metadata.reset_index()

model_path = '/Users/kaelynnrose/Documents/DATA_SCIENCE/projects/Earthquake_Detection/models/cnn_classification_50epochs_1738169508.keras'
model = tf.keras.models.load_model(model_path)
model.export('../models/1')

docker run -p 8501:8501 --name=earthquake-detection-model \
  --mount type=bind,source=/Users/kaelynnrose/Documents/DATA_SCIENCE/projects/Earthquake_Detection/models/model,destination=/models/model \
  -e MODEL_NAME=model -t tensorflow/serving


signal = raw_signals[0][:,2]

img = DataPreprocessing.plot_spectrogram(signal)
img = Image.fromarray(img).resize((100,150))
img = np.array(img) / 255.0

imgs = np.expand_dims(img, axis=0)

imgs = []
for i in range(0,1):
  signal = raw_signals[i][:,2]
  img = DataPreprocessing.plot_spectrogram(signal)
  img = Image.fromarray(img).resize((100,150))
  img = np.array(img) / 255.0
  imgs.append(img)
imgs = np.squeeze(np.array(imgs), axis=(0,)).tolist()

signal = raw_signals[i][:,2]
img = DataPreprocessing.plot_spectrogram(signal)
img = Image.fromarray(img).resize((100,150))
img = np.array(img) / 255.0
img = img.tolist()


data =  {
    "signature_name": "serving_default",  # Change this if you use a custom signature
    "instances": [{"input_layer": img}]
}
endpoint = "http://localhost:8501/v1/models/model:predict"
headers = {"Content-Type": "application/json"}

try:
    response = requests.post(endpoint, json=data, headers=headers)
    response.raise_for_status()  # Raise an exception if the response is an error
    prediction = response.json()['predictions'][0]  # Extract the prediction from the response
    return prediction
except requests.exceptions.RequestException as e:
    raise HTTPException(status_code=500, detail=f"Error connecting to TensorFlow Serving: {e}")


(np.array(prediction) > 0.5).astype(int)