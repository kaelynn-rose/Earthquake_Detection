

import json
import requests

import boto3
import matplotlib.pyplot as plt
import numpy as np

from io import BytesIO


s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')

def get_trace_from_s3(key, bucket_name):
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    file_stream = BytesIO(response['Body'].read())
    return np.load(file_stream)

def plot_results(trace, class_pred, earthquake_magnitude, bucket_name, key):
    fig, ax = plt.subplots(figsize=(6,4))
    ax.specgram(trace, Fs=100, NFFT=256, cmap='gray', vmin=-10, vmax=20); # plot spectrogram
    ax.set_xlim([0,60])
    ax.axis('off')
    plt.gca().set_axis_off() # remove axes
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    if class_pred == 'earthquake':
        plt.suptitle(f'Prediction: {class_pred}, Magnitude:{earthquake_magnitude}', fontweight='bold', fontsize=14)
    else:
        plt.suptitle(f'Prediction: {class_pred}, Magnitude: N/A', fontweight='bold', fontsize=14)
    ax.set_title(key, fontsize=8, wrap=True)
    plt.tight_layout()
    plt.show()
    fig_filename = f'{key}_{class_pred}.png'
    s3_filepath = f'live_data_prediction_images/{fig_filename}'
    plt.savefig(fig_filename, bbox_inches='tight')
    s3_resource.meta.client.upload_file(fig_filename, bucket_name, s3_filepath) # upload spectrogram and predictions to s3 bucket

def lambda_handler(event, context):
    bucket_name = event['Records'][0]['s3']['bucket']['name'] # get bucket name
    key = event['Records'][0]['s3']['object']['key'] # get event key

    trace = get_trace_from_s3(key, bucket_name)
    request = {'signal': trace.tolist(), 'sampling_rate':100}

    # Make request to earthquake detection API predict endpoint
    response = requests.post(
      'http://load-balancer-748306986.us-east-1.elb.amazonaws.com/predict',
        json.dumps(request)
    )

    if response.status_code == 200:
        class_pred = response.json()['signal_class_prediction']
        pred_prob = response.json()['signal_class_probability']
        earthquake_magnitude = response.json()['earthquake_magnitude_prediction']
        print(f'Predictions for trace {key}: class: {class_pred}, class probability: {pred_prob}, earthquake magnitude: {earthquake_magnitude}')
    else:
        print(f'Error getting results from earthquake-detection API: {requests.text}')

    plot_results(trace, class_pred, earthquake_magnitude, bucket_name, key)