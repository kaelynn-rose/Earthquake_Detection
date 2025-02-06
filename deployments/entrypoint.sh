#!/bin/bash

source activate earthquake-detection-serving

# Start Tensorflow Serving for signal classification model (earthquake vs. noise)
nohup tensorflow_model_server --rest_api_port=8501 --model_name=model_1 --model_base_path=/models/model_1 &

# Start Tensorflow Serving for earthquake magnitude prediction model
nohup nohup tensorflow_model_server --rest_api_port=8502 --model_name=model_2 --model_base_path=/models/model_2 &

# Run FastAPI application using Uvicorn
nohup uvicorn serving.api:app --host 0.0.0.0 --port 5000