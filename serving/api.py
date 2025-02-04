import logging
import sys
sys.path.append('../')

from fastapi import FastAPI, HTTPException

from serving import conf, proc

logger = logging.getLogger('earthquake-detection-api')

VERSION = '0.0.1'


app = FastAPI(
    title = 'Earthquake Detection API',
    description = (
        'This API provides an interface to ML models for prediction of seismic signal '
        'class (earthquake or noise) and earthquake magntiude prediction.'
    ),
    version=VERSION
)

@app.get('/earthquake-detection/', response_model=conf.HealthCheck)
@app.get('/earthquake-detection', response_model=conf.HealthCheck, include_in_schema=False)
@app.get('/', response_model=conf.HealthCheck, include_in_schema=False)
async def health_check():
    """Surfaces the result of the backend tensorflow serving model status check.
    This tensorflow model is the artifact that provides predictions."""
    return {
        'model_version_status': [
            'version': VERSION,
            'state': 'AVAILABLE',
            'status': 'OK',
            'error_message': ''
        ]
    }


@app.post('/earthquake-detection/predict')
@app.post('/predict')
def predict(request: conf.PredictionRequest):
    results = proc.EarthquakeDetection(request).get_predictions()
    return results