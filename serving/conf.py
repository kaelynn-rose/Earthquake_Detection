from typing import List, Optional, Union

from pydantic import BaseModel, Field

# Tensorflow Serving constants
TF_SERVING_ENDPOINTS = {
    'classification': {
        'health': 'http://localhost:8501/v1/models/classification-model',
        'predict': 'http://localhost:8501/v1/models/classification-model:predict'
    },
    'magnitude': {
        'health': 'http://localhost:8502/v1/models/magnitude-model',
        'predict': 'http://localhost:8502/v1/models/magnitude-model:predict'
    }
}
HEADERS = {"Content-Type": "application/json"}


class HealthCheck(BaseModel):
    status: str = Field(description='Overall status of all models')
    details: dict = Field(description='Model availability status details')

    class Config:
        json_encoders = {
            dict: lambda v: {k: str(v) for k, v in v.items()}  # Serialize status dictionary
        }

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





# model_status_dict = {}
# async with httpx.AsyncClient() as client:
#     for model_name in TF_SERVING_ENDPOINTS:
#         response = await client.get(TF_SERVING_ENDPOINTS[model_name]['health'])
#         response.raise_for_status() # Raises HTTPError if status code is not 200
#         model_status = response.json()
#         model_status_dict[model_name] = model_status
# print(model_status_dict)