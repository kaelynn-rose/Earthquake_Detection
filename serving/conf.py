from typing import List, Optional, Union

from pydantic import BaseModel, Field

# Tensorflow Serving constants
CLASSIFICATION_ENDPOINT = 'http://localhost:8501/v1/models/classification-model:predict'
MAGNITUDE_ENDPOINT = 'http://localhost:8502/v1/models/magnitude-model:predict'
HEADERS = {"Content-Type": "application/json"}




class PredictionRequest(BaseModel):
    """Validation model for the incoming request payload."""
    signal: List[float] = Field(
        description= (
            'A list of floating-point numbers representing the seismic signal data.'
            'Signals must be 6000 samples long (60 seconds at 100 Hz).'
        )
    )
    sampling_rate: Optional[int] = Field(
        default=100,
        description='Sampling rate of the signal, used for preprocessing before prediction'
    )

    class Config:
        schema_extra = {
            'example': {
                'signal': (

                )
            }
        }

class PredictionResponse(BaseModel):
    """Pydantic model for the response payload containing the model's predictions."""
    class_prediction: Union[str, None] = Field(
        description='The predicted class of the signal (earthquake or noise).'
    )
    status: str = Field(description='Prediction result status.')
    message: Union[str, None] = Field(description='Error message, if an error occurred.')