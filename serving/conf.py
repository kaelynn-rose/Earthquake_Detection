from typing import List, Optional, Union

from pydantic import BaseModel, Field

CLASSIFICATION_ENDPOINT = 'http://localhost:8501/v1/models/classification-model:predict'
MAGNITUDE_ENDPOINT = 'http://localhost:8502/v1/models/magnitude-model:predict'
HEADERS = {"Content-Type": "application/json"}


class PredictionRequest(BaseModel):
    """Validation model for the incoming request payload."""
    signal: List[float] = Field(
        description='A list of floating-point numbers representing the seismic signal data.'
    )
    sampling_rate: Optional[int] = Field(
        default=100,
        description='Sampling rate of the signal, used for preprocessing before prediction'
    )

class PredictionResponse(BaseModel):
    """Pydantic model for the response payload containing the model's predictions."""
    class_prediction: Union[str, None] = Field(
        description='The predicted class of the signal (earthquake or noise).'
    )
    status: str = Field(description='Prediction result status.')
    message: Union[str, None] = Field(description='Error message, if an error occurred.')