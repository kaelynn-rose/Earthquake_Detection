


import requests
import sys

sys.path.append('../')
sys.path.append('../../../')

import numpy as np
import pandas as pd

from FastAPI import HTTPException
from PIL import Image

import earthquake_detection.data_preprocessing as DataPreprocessing
import serving.conf as conf


# # Load extracted raw signals
# raw_signals = np.load('../../../data/STEAD/extracted_raw_signals_subsample_1000.npy')

# # Load metadata
# metadata = pd.read_feather('../../../data/STEAD/extracted_metadata_subsample_1000.feather')
# metadata = metadata.reset_index()

# model_path = '/Users/kaelynnrose/Documents/DATA_SCIENCE/projects/Earthquake_Detection/models/cnn_classification_50epochs_1738169508.keras'
# model = tf.keras.models.load_model(model_path)
# model.export('../models/1')

# docker run -p 8501:8501 --name=earthquake-detection-model \
#   --mount type=bind,source=/Users/kaelynnrose/Documents/DATA_SCIENCE/projects/Earthquake_Detection/models/model,destination=/models/model \
#   -e MODEL_NAME=model -t tensorflow/serving




# signal = raw_signals[i][:,2]
# img = DataPreprocessing.plot_spectrogram(signal)
# img = Image.fromarray(img).resize((100,150))
# img = np.array(img) / 255.0

# # Use this for prediction straight from model artifact
# imgs = np.expand_dims(img, axis=0)

# # Use this for prediction with model served with tensorflow serving
# img = img.tolist()


# data =  {
#     "signature_name": "serving_default",  # Change this if you use a custom signature
#     "instances": [{"input_layer": img}]
# }
# endpoint = "http://localhost:8501/v1/models/model:predict"
# headers = {"Content-Type": "application/json"}

# try:
#     response = requests.post(endpoint, json=data, headers=headers)
#     response.raise_for_status()  # Raise an exception if the response is an error
#     prediction = response.json()['predictions'][0]  # Extract the prediction from the response
#     return prediction
# except requests.exceptions.RequestException as e:
#     raise HTTPException(status_code=500, detail=f"Error connecting to TensorFlow Serving: {e}")


# (np.array(prediction) > 0.5).astype(int)



class GetPredictions():
    def __init__(request):
        self.signal = request.signal
        self.sampling_rate = request.sampling_rate
        self.results = {}

    def preproc(self, img_size):
        img = DataPreprocessing.plot_spectrogram(self.signal[:,2], self.sampling_rate)
        img = Image.fromarray(img).resize(img_size) # Resize to the size the chosen model accepts
        img_arr = np.array(img) / 255.0
        self.preproc_img = img_arr.tolist()

    def get_classification_prediction(self, img_arr):
        data =  {
            "signature_name": "serving_default",
            "instances": [{"input_layer": self.preproc_img}]
        }
        self.class_pred_prob = self.get_prediction_from_tf_serving(
            endpoint=conf.CLASSIFICATION_ENDPOINT,
            data=data,
            headers=conf.HEADERS
        )
        self.class_pred = 'earthquake' if self.class_pred_prob[0] > 0.5 else 'noise'

    def get_magnitude_prediction(self):
        data =  {
            "signature_name": "serving_default",
            "instances": [{"input_layer": self.preproc_img}]
        }
        self.magnitude_pred = self.get_prediction_from_tf_serving(
            endpoint=conf.MAGNITUDE_ENDPOINT,
            data=data,
            headers=conf.HEADERS
        )
        #TODO add postproc
        pass

    def get_prediction_from_tf_serving(self, endpoint, headers, data):
        try:
            response = requests.post(
                endpoint,
                json=data,
                headers=headers
            )
            response.raise_for_status()  # Raise an exception if the response is an error
            prediction = response.json()['predictions'][0]
            return prediction
        except requests.exceptions.RequestException as e:
            raise HTTPException(
                status_code=500,
                detail=f'Error connecting to TensorFlow Serving: {e}'
            )

    def get_predictions(self):
        classification_img = self.preproc(img_size=(150,100))
        self.get_classification_prediction(classification_img)
        if self.class_pred == 'earthquake':
            magnitude_img = self.preproc(img_size=(300,100))
            self.get_magnitude_prediction(magnitude_img)
        else:
            self.magnitude_pred = None
        predictions = {
            'signal_class': self.class_pred,
            'signal_class_probability': self.class_pred_prob,
            'earthquake_magnitude': self.magnitude_pred
        }