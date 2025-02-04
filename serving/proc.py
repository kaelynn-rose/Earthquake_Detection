

import logging
import requests
import sys

sys.path.append('../')
sys.path.append('../../../')

import httpx
import numpy as np
import pandas as pd

from PIL import Image

import earthquake_detection.data_preprocessing as DataPreprocessing
import serving.conf as conf

logger = logging.getLogger('earthquake-detection-api')


# # Load extracted raw signals
# raw_signals = np.load('../../../data/STEAD/extracted_raw_signals_subsample_1000.npy')

# # Load metadata
# metadata = pd.read_feather('../../../data/STEAD/extracted_metadata_subsample_1000.feather')
# metadata = metadata.reset_index()

# model_path = '/Users/kaelynnrose/Documents/DATA_SCIENCE/projects/Earthquake_Detection/models/cnn_classification_50epochs_1738169508.keras'
# model = tf.keras.models.load_model(model_path)
# model.export('../signal_classification_model/1')

# model_path = '/Users/kaelynnrose/Documents/DATA_SCIENCE/projects/Earthquake_Detection/models/cnn_regression_magnitude_20epochs_1738185565.keras'
# model = tf.keras.models.load_model(model_path)
# model.export('../earthquake_magnitude_prediction_model/1')

# model_path = '/Users/kaelynnrose/Documents/DATA_SCIENCE/projects/Earthquake_Detection/models/earthquake_magnitude_prediction_model/1'
# model = tf.saved_model.load(model_path)


# docker run -p 8501:8501 --name=earthquake-detection-model \
#   --mount type=bind,source=/Users/kaelynnrose/Documents/DATA_SCIENCE/projects/Earthquake_Detection/models/model,destination=/models/model \
#   -e MODEL_NAME=model -t tensorflow/serving

# docker run -p 8501:8501 --name=classification-model \
#   --mount type=bind,source=/Users/kaelynnrose/Documents/DATA_SCIENCE/projects/Earthquake_Detection/models/signal_classification_model,destination=/models/classification-model \
#   -e MODEL_NAME=classification-model -t tensorflow/serving

# docker run -p 8502:8501 --name=magnitude-model \
#   --mount type=bind,source=/Users/kaelynnrose/Documents/DATA_SCIENCE/projects/Earthquake_Detection/models/earthquake_magnitude_prediction_model,destination=/models/magnitude-model \
#   -e MODEL_NAME=magnitude-model -t tensorflow/serving


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
#     "instances": [{"input_layer_1": img}]
# }
# endpoint = MAGNITUDE_ENDPOINT
# headers = {"Content-Type": "application/json"}

# try:
#     response = requests.post(endpoint, json=data, headers=headers)
#     response.raise_for_status()  # Raise an exception if the response is an error
#     prediction = response.json()['predictions'][0]  # Extract the prediction from the response
#     return prediction
# except requests.exceptions.RequestException as e:
#     raise HTTPException(status_code=500, detail=f"Error connecting to TensorFlow Serving: {e}")


# (np.array(prediction) > 0.5).astype(int)



class EarthquakeDetection():
    def __init__(self, request):
        self.signal = request.signal
        #self.signal = request['signal']
        self.sampling_rate = request.sampling_rate
        #self.sampling_rate = request['sampling_rate']
        self.results = {}
        self.status = 'OK'
        self.message = ''

    def preproc(self, img_size):
        img = DataPreprocessing.plot_spectrogram(self.signal, self.sampling_rate)
        img = Image.fromarray(img).resize(img_size) # Resize to match the input size for the model
        img_arr = np.array(img) / 255.0
        return img_arr.tolist()

    def get_classification_prediction(self):
        self.classification_img = self.preproc(img_size=(100,150))
        data =  {
            "signature_name": "serving_default",
            "instances": [{"input_layer": self.classification_img}]
        }
        self.class_pred_prob = self.get_prediction_from_tf_serving(
            endpoint=conf.TF_SEVRVING_ENDPOINTS['classification']['predict'],
            data=data,
            headers=conf.HEADERS
        )
        self.class_pred = 'earthquake' if self.class_pred_prob[0] > 0.5 else 'noise'

    def get_magnitude_prediction(self):
        self.magnitude_img = self.preproc(img_size=(100,150))
        data =  {
            "signature_name": "serving_default",
            "instances": [{"input_layer_1": self.magnitude_img}]
        }
        magnitude_pred = self.get_prediction_from_tf_serving(
            endpoint=conf.TF_SEVRVING_ENDPOINTS['magnitude']['predict'],
            data=data,
            headers=conf.HEADERS
        )
        self.magnitude_pred = magnitude_pred[0]

    def get_prediction_from_tf_serving(self, endpoint, headers, data):
        try:
            response = requests.post(
                endpoint,
                json=data,
                headers=headers
            )
            print(response.text)
            response.raise_for_status()  # Raise an exception if the response is an error
            prediction = response.json()['predictions'][0]
            return prediction
        except requests.exceptions.RequestException as e:
            self.status = False
            self.message = f'Error connecting to TensorFlow Serving: {e}'
            logger.warning(self.message)
            return None

    def get_predictions(self):
        self.get_classification_prediction()
        if self.class_pred == 'earthquake':
            self.get_magnitude_prediction()
        else:
            self.magnitude_pred = None
        if self.status:
            self.results = {
                'signal_class_prediction': self.class_pred,
                'signal_class_probability': self.class_pred_prob,
                'earthquake_magnitude_prediction': self.magnitude_pred,
            }
        else:
            self.results = {
                'status': 'ERROR',
                'message': self.message,
                'signal_class_prediction': None,
                'signal_class_probability': None,
                'earthquake_magnitude_prediction': None
            }
        return self.results