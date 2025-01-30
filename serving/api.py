import logging
import sys
sys.path.append('../')

from fastapi import FastAPI, HTTPException

from earthquake_detection.serving import conf, proc

logger = logging.getLogger('earthquake-detection-api')

VERSION = '0.0.1'


app = FastAPI(
    title = 'Earthquake Detection API',
    description = 'TODO',
    version=VERSION
)

@app.get('/earthquake-detection/', response_model=conf.HealthCheck)
@app.get('/earthquake-detection', response_model=conf.HealthCheck, include_in_schema=False)
async def health_check():
    """Surfaces the result of the backend tensorflow serving model status check.
    This tensorflow model is the artifact that provides predictions."""
    return {
        'model_version_status'
    }