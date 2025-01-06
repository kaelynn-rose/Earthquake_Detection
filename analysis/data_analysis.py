

import pathlib

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR_PATH = '/Users/kaelynnrose/Documents/DATA_SCIENCE/data'
SUBSAMPLE_N = 50000 # the number of signal traces to sample from the full dataset

def is_populated(arr):
    # Check if array is empty or has only None (dtype=object)
    if arr.size == 0:  # Check if array is empty
        return False
    if arr.dtype == object and np.all(arr == None):  # Check if array contains only None values
        return False
    return True

def plot_signal_trace(trace):
    fig, ax = plt.subplots(figsize=(3,2))
    ax.specgram(tr[:,2], Fs=100, NFFT=256, cmap='gray', vmin=-10, vmax=25) # select only the z-axis component of the signals
    ax.set_xlim([0,60])
    ax.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    return fig

def plot_spectrogram_from_trace(trace):
    fig, ax = plt.subplots(figsize=(3,2))
    ax.plot(np.linspace(0,60,6000),tr[:,2],color='k',linewidth=1)
    ax.set_xlim([0,60])
    ax.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
    plt.margins(0,0)
    return fig

# Fetch data
metadata_paths, data_paths = [], []
data_dir = Path(DATA_DIR_PATH)

for file in data_dir.rglob('*.csv'):
    metadata_paths.append(file)
for file in data_dir.rglob('*.hdf5'):
    data_paths.append(file)

# Parse metadata from csv files
metadata_dfs = []
for i, path in enumerate(metadata_paths):
    df = pd.read_csv(path)
    df['chunk'] = i+1 # persist info on which trace came from which data chunk
    metadata_dfs.append(df)

# Combine metadata into one dataframe
metadata_df = pd.concat(metadata_dfs).reset_index(drop=True)

print(
    f'Number of total traces: {len(metadata_df)}\n'
    f'Number of earthquake traces: {len(metadata_df[metadata_df['trace_category'].str.contains('earthquake')])}\n'
    f'Number of noise traces: {len(metadata_df[metadata_df['trace_category']=='noise'])}'
)

# Fetch subsample of traces from hdf5 files
traces_df = metadata_df.sample(SUBSAMPLE_N, random_state=0)
trace_names = traces_df['trace_name']

traces = {}
for i, path in enumerate(data_paths):
    print(f'Parsing traces from file at path {i}/{len(data_paths)}')
    h5f = h5py.File(path, mode='r')
    for _, trace_name in enumerate(tqdm(trace_names)):
        try:
            trace = np.array(h5f.get(f'data/{trace_name}'))
            if is_populated(trace):
                traces[trace_name] = trace
        except Exception as e:
            pass

imgs = []
keys = []
for key, val in traces.items():


