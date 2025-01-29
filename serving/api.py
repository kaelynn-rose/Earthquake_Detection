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