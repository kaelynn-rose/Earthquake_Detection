'''Data preprocessing module for earthquake detection using the STanford EArthquake
Dataset (STEAD) seismic signal dataset.

**This script requires that you have downloaded all data chunks from STEAD to your
local machine. The data can be downloaded here: https://github.com/smousavi05/STEAD.

This module contains functions that:
1. Fetch the STEAD data from its locally saved filepaths
2. Parses the seismic signal metadata from the csv files
3. Reads a random subsample of the seismic traces from the .hdf5 formatted data files
4. Transforms the seismic signals into spectrogram (a visual representation of the
   spectrum of frequencies of a signal as it varies with time) plots
5. Saves the spectrogram plots to a np.array to be used as model training data

Additionally, this module contains utilities for converting the seismic signal traces
into either waveform plots or spectrogram plots, for analysis purposes.
'''


from io import BytesIO
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

DATA_DIR_PATH = '/Users/kaelynnrose/Documents/DATA_SCIENCE/data'
SUBSAMPLE_N = 1000 # the number of signal traces to sample from the full dataset

class DataPreprocessing():
    def __init__(self, DATA_DIR_PATH):
        self.data_dir_path = DATA_DIR_PATH

    def _is_populated(self, arr):
        '''Check if array is empty or has only None (dtype=object).

        Parameters
        ----------
        arr : np.array()
            The array to check if empty or None

        Returns
        -------
        bool : Whether the array is empty'''
        if arr.size == 0:  # Check if array is empty
            return False
        if arr.dtype == object and np.all(arr == None):  # Check if array contains only None values
            return False
        return True

    def plot_spectrogram(self, trace):
        '''Plots a spectrogram image given a seismic signal trace.

        Parameters
        ----------
        trace : np.array()
            The raw seismic signal array to plot the spectrogram image of

        Returns
        -------
        matplotlib.pyplot figure showing the spectrogram of the signal trace'''
        fig, ax = plt.subplots(figsize=(3,2))
        ax.specgram(trace[:,2], Fs=100, NFFT=256, cmap='gray', vmin=-10, vmax=25) # select only the z-axis component of the signals
        ax.set_xlim([0,60])
        ax.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        return fig

    def plot_waveform(self, trace):
        '''Plots a waveform image given a seismic signal trace.

        Parameters
        ----------
        trace : np.array()
            The raw seismic signal array to plot the waveform image of

        Returns
        -------
        matplotlib.pyplot figure showing the waveform of the signal trace'''
        fig, ax = plt.subplots(figsize=(3,2))
        ax.plot(np.linspace(0,60,6000), trace[:,2], color='k', linewidth=1)
        ax.set_xlim([0,60])
        ax.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0,0)
        return fig

    def _fetch_datapaths_from_dir(self):
        '''Finds .csv and .hdf5 filepaths within any subdirectories of the data
        directory path specified when the class is initiated (see class init), and
        appends them to lists'''
        print('Fetching data paths from directory')
        self.metadata_paths = []
        self.data_paths = []
        data_dir = Path(self.data_dir_path)
        for file in data_dir.rglob('*.csv'):
            self.metadata_paths.append(file)
        for file in data_dir.rglob('*.hdf5'):
            self.data_paths.append(file)

    def _parse_metadata_csvs(self):
        '''Reads metadata from the individual STEAD dataset csv files and
        combines it into a single dataframe.'''
        print('Parsing metadata from csv files')
        metadata_dfs = []
        for i, path in enumerate(self.metadata_paths):
            df = pd.read_csv(path)
            df['chunk'] = i+1 # persist info on which trace came from which data chunk
            metadata_dfs.append(df)

        # Combine metadata into one dataframe
        self.full_metadata = pd.concat(metadata_dfs).reset_index(drop=True)

        num_total = len(self.full_metadata)
        num_earthquakes = len(self.full_metadata[self.full_metadata['trace_category'].str.contains('earthquake')])
        num_noise = len(self.full_metadata[self.full_metadata['trace_category']=='noise'])
        print(
            f'Number of total traces: {num_total}\n'
            f'Number of earthquake traces: {num_earthquakes}\n'
            f'Number of noise traces: {num_noise}'
        )

    def _get_traces_subsample(self, subsample_n):
        '''Collects a subsample of the seismic traces from the full STEAD dataset.

        Parameters
        ----------
        subsample_n : int
            The number of signal traces to randomly sample from the full dataset

        Returns
        -------
        pd.DataFrame containing metadata for the subsample of signal traces'''
        print('Fetching subsample of traces from hdf5 files')
        subsample_metadata = self.full_metadata.sample(subsample_n, random_state=0)
        return subsample_metadata

    def _read_h5py_files(self, trace_names):
        traces = {}
        for i, path in enumerate(self.data_paths):
            print(f'Parsing traces from file at path # {i}/{len(self.data_paths)}')
            h5f = h5py.File(path, mode='r')
            for _, trace_name in enumerate(tqdm(trace_names)):
                try:
                    trace = np.array(h5f.get(f'data/{trace_name}'))
                    if self._is_populated(trace):
                        traces[trace_name] = trace
                except Exception as e:
                    pass
        return traces

    def preprocess(self, subsample_n):
        print(f'Data preprocessing for subsample of signal data of size {subsample_n}')
        self._fetch_datapaths_from_dir()
        self. _parse_metadata_csvs()
        self.subsample_metadata = self._get_traces_subsample(subsample_n)
        subsample_trace_names = self.subsample_metadata['trace_name']
        self.subsample_traces = self._read_h5py_files(subsample_trace_names)

        # Convert the signal traces into spectrogram images and then store the images in an array
        self.subsample_imgs = []
        for trace_name, trace in tqdm(self.subsample_traces.items()):
            try:
                img = self.plot_spectrogram(trace)  # Convert the signal to a spectrogram image
                buf = BytesIO()
                img.savefig(buf, format='png') # Save the plot to a BytesIO buffer, to avoid saving images to disk
                buf.seek(0) # Rewind the buffer so we can read the contents from the start
                img = Image.open(buf) # Open the image from the buffer
                img_arr = np.array(img)
                self.subsample_imgs.append(img_arr)
                img.close()
            except Exception as e:
                print(
                    f'Unable to plot or save spectrogram array for trace {trace_name}.'
                    f'Dropping this trace from the dataset and metadata. \n Exception: {e}'
                )
                index_to_remove = self.subsample_metadata[self.subsample_metadata['trace_name'] == trace_name].index
                self.subsample_metadata = self.subsample_metadata.drop(index_to_remove)

        return self.subsample_traces, self.subsample_imgs, self.subsample_metadata


preproc = DataPreprocessing(DATA_DIR_PATH)
raw_signals, imgs, metadata = preproc.preprocess(subsample_n=SUBSAMPLE_N)