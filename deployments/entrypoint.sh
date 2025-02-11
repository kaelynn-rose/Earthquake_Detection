#!/bin/bash

source activate earthquake-detection-serving

# Start Tensorflow Serving for signal classification model (earthquake vs. noise)
nohup tensorflow_model_server --rest_api_port=8501 --model_name=classification-model --model_base_path=/home/app/models/model_1 &

# Start Tensorflow Serving for earthquake magnitude prediction model
nohup nohup tensorflow_model_server --rest_api_port=8502 --model_name=magnitude-model --model_base_path=/home/app/models/model_2 &

# Run FastAPI application using Uvicorn on port 5000
uvicorn serving.api:app --host 0.0.0.0 --port 5000