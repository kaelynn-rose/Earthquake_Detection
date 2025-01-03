
import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


S3_BUCKET = 'earthquake-detection'
S3_DATA_PATH = 'data/'

s3 = boto3.client('s3')

# Fetch data from S3
response = s3.list_objects_v2(Bucket=S3_BUCKET, Prefix=S3_DATA_PATH)

metadata_paths = []
data_paths = []
if 'Contents' in response:
    for i, obj in enumerate(response['Contents']):
        if '.csv' in obj['Key']:
            metadata_paths.append('s3://' + S3_BUCKET + '/' + obj['Key'])
        elif '.hdf5' in obj['Key']:
            data_paths.append('s3://' + S3_BUCKET + '/' + obj['Key'])

# Parse metadata from csv files
metadata_dfs = [pd.read_csv(path) for path in metadata_paths]
metadata_df = pd.concat(metadata_dfs)
print(
    f'Number of total traces: {len(metadata_df)}\n'
    f'Number of earthquake traces: {len(metadata_df[metadata_df['trace_category'].str.contains('earthquake')])}\n'
    f'Number of noise traces: {len(metadata_df[metadata_df['trace_category']=='noise'])}'
)


