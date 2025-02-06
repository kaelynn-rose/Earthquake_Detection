"""Defines API utilities for preprocessing, requesting predictions from models
served with Tensorflow Serving, and postprocessing."""

import logging
import requests
import sys

sys.path.append('../')
sys.path.append('../../../')

import numpy as np

from PIL import Image

import earthquake_detection.data_preprocessing as DataPreprocessing
import serving.conf as conf

logger = logging.getLogger('earthquake-detection-api')
logging.basicConfig(level=logging.INFO)

class EarthquakeDetection():
    """Class that defines the flow for signal processing, model prediction, and
    postprocessing steps.

    Parameters
    ----------
    request : conf.PredictionRequest
        A PredictionRequest object. See conf.py for details."""
    def __init__(self, request):
        self.signal = request.signal
        self.sampling_rate = request.sampling_rate
        self.results = {}
        self.status = 'OK'
        self.message = ''

        logger.info('Starting detection class')

    def preproc(self, img_size):
        """Utility for preprocessing seismic signals for input into ML models.
        Takes in the seismic signal array, plots it as a spectrogram, converts
        the spectrogram into an array, resizes and normalizes the array, and then
        returns as a list for input into the RESTful APIs exposed by TF Serving.

        Parameters
        ----------
        img_size : tuple
            The dimensions of the image that match the input size for the model.
            For example, img_size=(100,150) for a model that accepts images of
            height=100 and width=150.

        Returns
        -------
        The preprocessed image array in list format."""
        logger.info('Preprocessing seismic signal to image array for model input.')
        img = DataPreprocessing.plot_spectrogram(self.signal, self.sampling_rate)
        img = Image.fromarray(img).resize(img_size) # Resize to match the input size for the model
        img_arr = np.array(img) / 255.0
        return img_arr.tolist()

    def get_classification_prediction(self):
        """Implements preprocessing on the input signal, populates data for the
        TF Serving request, and sends the request to the classification model
        via the TF Serving RESTful API. Applies postprocessing to prediction results."""
        self.classification_img = self.preproc(img_size=(100,150))
        data =  {
            "signature_name": "serving_default",
            "instances": [{"input_layer": self.classification_img}]
        }
        logger.info('Requesting prediction from classification model.')
        self.class_pred_prob = self.get_prediction_from_tf_serving(
            endpoint=conf.TF_SERVING_ENDPOINTS['classification']['predict'],
            data=data,
            headers=conf.HEADERS
        )[0]
        self.class_pred = 'earthquake' if self.class_pred_prob > 0.5 else 'noise'

    def get_magnitude_prediction(self):
        """Implements preprocessing on the input signal, populates data for the
        TF Serving request, and sends the request to the earthquake magnitude model
        via the TF Serving RESTful API. Applies postprocessing to prediction results."""
        self.magnitude_img = self.preproc(img_size=(100,150))
        data =  {
            "signature_name": "serving_default",
            "instances": [{"input_layer_1": self.magnitude_img}]
        }
        logger.info('Requesting prediction from earthquake magnitude model.')
        magnitude_pred = self.get_prediction_from_tf_serving(
            endpoint=conf.TF_SERVING_ENDPOINTS['magnitude']['predict'],
            data=data,
            headers=conf.HEADERS
        )
        self.magnitude_pred = magnitude_pred[0]

    def get_prediction_from_tf_serving(self, endpoint, headers, data):
        """Sends a request to the TF Serving endpoint to get ML model predictions.

        Parameters
        ----------
        endpoint : str
            The URL for the endpoint of the TF Serving API for the model we would like to
            get predictions from.
            Example: http://localhost:8501/v1/models/classification-model:predict
        headers : dict
            Key-value pairs sent with the request to the TF Serving API to provide
            additional information bout the request being made.
            Example: {"Content-Type": "application/json"}
        data : dict
            The data to send in the request payload, containing the preprocessed
            seismic signal array input for the ML model to use for prediction."""
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
            self.status = False
            self.message = f'Error connecting to TensorFlow Serving: {e}'
            logger.warning(self.message)
            return None

    def get_predictions(self):
        """Requests predictions from the classification and magnitude ML models,
        and returns a results dictionary. Only provides earthquake magnitude prediction
        if the signal is predicted to be an earthquake by the classification model.

        Returns
        -------
        signal_class_prediction : str
            'earthquake' or 'noise' signal class prediction as predicted by the
            ML classification model.
        signal_class_probability : float
            The probability of the signal being an earthquake, as predicted by the
            ML classification model.
        earthquake_magnitude_prediction : float or None
            The earthquake magnitude if the signal is predicted to be an
            earthquake, else None.
        status : str
            The API status; 'OK' or 'ERROR' if models failed to return prediction results.
        message : str
            Error message, if any."""
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
                'status': 'OK',
                'message': '',
            }
        else:
            self.results = {
                'signal_class_prediction': None,
                'signal_class_probability': None,
                'earthquake_magnitude_prediction': None,
                'status': 'ERROR',
                'message': self.message
            }
        return self.results