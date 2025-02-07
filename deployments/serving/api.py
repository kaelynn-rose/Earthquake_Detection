"""This API uses two machine learning models to predict 1) whether a seismic signal
payload is an earthquake or noise, and 2) the magnitude of the signal if the signal
is predicted to be an earthquake. This module implements signal preprocessing,
interfaces with the ML models served using Tensorflow Serving, applies postprocessing
steps, and returns prediction results. It provides an interface with the Uvicorn
server gateway for running the API."""

import logging
import sys

sys.path.append('../')

import httpx

from fastapi import FastAPI

from serving import conf, proc

logger = logging.getLogger('earthquake-detection-api')

VERSION = '0.0.1'


app = FastAPI(
    title = 'Earthquake Detection API',
    description = (
        'This API provides an interface to ML models for prediction of seismic signal '
        'class (earthquake or noise) and earthquake magntiude prediction.'
    ),
    version = VERSION
)

@app.get('/earthquake-detection/', response_model=conf.HealthCheck)
@app.get('/earthquake-detection', response_model=conf.HealthCheck, include_in_schema=False)
@app.get('/', response_model=conf.HealthCheck, include_in_schema=False)
async def health_check():
    """Surfaces the result of the backend tensorflow serving model status check.
    This tensorflow model is the artifact that provides predictions."""
    model_status_dict = {}
    overall_status = 'HEALTHY'
    async with httpx.AsyncClient() as client:
        for model_name in conf.TF_SERVING_ENDPOINTS:
            try:
                response = await client.get(conf.TF_SERVING_ENDPOINTS[model_name]['health'])
                model_status = response.json()
                model_status_dict[model_name] = model_status
                response.raise_for_status()
            except Exception as e:
                overall_status = 'UNHEALTHY'
    return conf.HealthCheck(status=overall_status, details=model_status_dict)

@app.post('/earthquake-detection/predict', response_model=conf.PredictionResponse)
@app.post('/predict', response_model=conf.PredictionResponse, include_in_schema=False)
def predict(request: conf.PredictionRequest):
    """Main prediction endpoint for the classification and earthquake magnitude
    ML models. Takes an input seismic signal of length 6000 samples (60 seconds of
    signal at 100 Hz sampling rate) and predicts 1) whether the signal is an
    earthquake or noise, and 2) if the signal is predicted to be an earthquake,
    predicts the earthquake magnitude.

    Parameters
    ----------
    request : conf.PredictionRequest
        A PredictionRequest object

    Returns
    -------
    A conf.PredictionResponse object which contains predictions for signal class
    and earthquake magnitude if applicable"""
    logger.debug('hello world')
    results = proc.EarthquakeDetection(request).get_predictions()
    logger.info(f'Returning results {results}')
    return results