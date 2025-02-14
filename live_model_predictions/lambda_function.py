"""Lambda function triggered by uploading of a .npy object to the specified S3 bucket.
The Lambda function calls the earthquake-detection API (code for the API can be
found in the /deployments/serving directory of this repository) and returns ML model
predictions for each uploaded signal object."""

import json
import requests

import boto3
import matplotlib.pyplot as plt
import numpy as np

from io import BytesIO

plt.ioff() # Turn off matplotlib interactive mode to ensure plots are clearing from memory. Prevents  memory leakage.

s3_client = boto3.client('s3')
s3_resource = boto3.resource('s3')


def get_trace_from_s3(key, bucket_name):
    """Fetch seismic trace from S3 bucket

    Paramters
    ---------
    key : str
        The S3 filepath of the trace to fetch
    bucket_name : str
        The S3 bucket to fetch the trace from

    Returns
    -------
    A np.ndarray object containing the seismic trace array"""
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    file_stream = BytesIO(response['Body'].read())
    return np.load(file_stream)

def plot_results(trace, class_pred, earthquake_magnitude, bucket_name, key):
    """Plots the spectrogram of the seismic trace and titles the plot with the
    model predictions.

    Paramters
    ---------
    trace : np.ndarray
        The seismic trace to convert to a spectrogram to plot
    class_pred : str
        Signal class prediction (either 'earthquake' or 'noise')
    earthquake_magnitude : float
        Magnitude prediction, if the signal is predicted to be an earthquake
    bucket_name : str
        The S3 bucket to fetch the trace from
    key : str
        The S3 filepath of the trace to fetch"""
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
    """Lambda function that processes an event, triggered when an item is
    added to the S3 bucket.

    Parameters
    ----------
    event : dict
        The event data passed by the Lambda trigger
    context : LambdaContext
        The context object containing runtime information"""
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
        print(f'Error getting results from earthquake-detection API: {response.text}')

    plot_results(trace, class_pred, earthquake_magnitude, bucket_name, key)
    return response.json()