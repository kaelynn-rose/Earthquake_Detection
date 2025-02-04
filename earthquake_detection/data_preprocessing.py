'''Data preprocessing module for earthquake detection using the STanford EArthquake
Dataset (STEAD) seismic signal dataset.

**This script requires that you have downloaded all data chunks from STEAD to your
local machine. The data can be downloaded here: https://github.com/smousavi05/STEAD.

This module contains functions that:
1. Fetch the STEAD data from its locally saved filepaths
2. Parses the seismic signal metadata from the csv files
3. Reads a random subsample of the seismic traces from the .hdf5 formatted data files
4. Transforms the seismic signals into waveform and spectrogram (a visual representation
   of the spectrum of frequencies of a signal as it varies with time) plots
5. Saves the waveform and spectrogram plots to a np.array to be used as model training data

Additionally, this module contains utilities for converting the seismic signal traces
into either waveform plots or spectrogram plots, for analysis purposes.

Examples
--------
> preproc = DataPreprocessing(YOUR_DATA_DIR_PATH_HERE, SUBSAMPLE_N=5000)
> raw_signals = preproc.subsample_traces # access raw signals
> metadata = preproc.subsample_metadata # access metadata
> waveform_imgs = preproc.create_waveform_images(img_width=3, img_height=2, img_dpi=100)
> spectrogram_imgs = preproc.create_spectrogram_images(img_width=3, img_height=2, img_dpi=100) '''

import gc

from io import BytesIO
from pathlib import Path

import h5py
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from PIL import Image
from tqdm import tqdm

matplotlib.use('Agg') # sets Matplotlib to generate images without the need for a graphical interface; avoids GUI related errors with FastAPI


def plot_spectrogram(trace, sampling_rate=100, img_width=3, img_height=2, dpi=100):
    '''Plots a spectrogram image for the Z-axis of a seismic signal trace, given a
    numpy.ndarray with 3 columns, the X, Y, and Z axes.

    Parameters
    ----------
    trace : np.array()
        The raw seismic signal array to plot the spectrogram image of

    Returns
    -------
    matplotlib.pyplot figure showing the spectrogram of the signal trace'''
    fig, ax = plt.subplots(figsize=(img_width,img_height), dpi=dpi)
    ax.specgram(trace, Fs=sampling_rate, NFFT=256, cmap='gray', vmin=-10, vmax=25)
    ax.set_xlim([0,60])
    ax.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)

    buf = BytesIO()
    fig.savefig(buf, format='png') # Save the plot to a BytesIO buffer, to avoid saving images to disk
    buf.seek(0) # Rewind the buffer so we can read the contents from the start
    img = Image.open(buf) # Open the image from the buffer
    img_arr = np.array(img)
    if img_arr.shape[2] == 4:
        img_arr = img_arr[:,:,:3] # drop alpha channel if there is one
    plt.close(fig)
    plt.ioff()
    del img
    del fig
    del buf

    return img_arr

def plot_waveform(trace, img_width=3, img_height=1, dpi=100):
    '''Plots a waveform image for the Z-axis of a seismic signal trace, given a
    numpy.ndarray with 3 columns, the X, Y, and Z axes.

    Parameters
    ----------
    trace : numpy.ndarray
        The raw seismic signal array to plot the waveform image of

    Returns
    -------
    matplotlib.pyplot figure showing the waveform of the signal trace'''
    fig, ax = plt.subplots(figsize=(img_width,img_height), dpi=dpi)
    x = np.linspace(0,60,6000)
    ax.plot(x, trace, color='k', linewidth=1)
    ax.set_xlim([0,60])
    ax.axis('off')
    plt.gca().set_axis_off()
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0,0)

    buf = BytesIO()
    fig.savefig(buf, format='png') # Save the plot to a BytesIO buffer, to avoid saving images to disk
    buf.seek(0) # Rewind the buffer so we can read the contents from the start
    img = Image.open(buf) # Open the image from the buffer
    img_arr = np.array(img)
    if img_arr.shape[2] == 4:
        img_arr = img_arr[:,:,:3] # drop alpha channel if there is one
    plt.close(fig)
    plt.ioff()
    del img
    del fig
    del buf

    return img_arr

class DataPreprocessing():
    '''A class to perform data preprocessing to fetch data from files of the
    STEAD seismic signal dataset and convert a subsample of signal traces to
    spectrograms for ML model training. See module docstring above.'''
    def __init__(self, data_dir_path, subsample_n, weighted):
        '''Initializes the DataPreprocessing object

        Parameters
        ----------
        data_dir_path : str
            The local directory to search for the STEAD dataset csv and h5py files
        subsample_n : int
            The number of traces to randomly sample from the complete dataset (all
            of the h5py files contained in data_dir_path)
        weighted : bool
            Whether to weight the likelihood of rows being selected to the random
            sample based on the total number of each label in the dataset'''
        self.data_dir_path = data_dir_path
        self.subsample_n = subsample_n
        self.weighted = weighted

        self._fetch_datapaths_from_dir()
        self. _parse_metadata_csvs()
        self.subsample_metadata = self._get_traces_subsample()
        subsample_trace_names = list(self.subsample_metadata.index)
        self.subsample_traces = self._read_h5py_files(subsample_trace_names)

        new_metadata_order = self.subsample_traces.keys()
        self.subsample_metadata = self.subsample_metadata.loc[new_metadata_order]

        plt.ioff()

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
        for i, path in enumerate(tqdm(self.metadata_paths)):
            df = pd.read_csv(path, low_memory=False)
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

    def _get_traces_subsample(self):
        '''Collects a subsample of the seismic traces from the full STEAD dataset.

        Parameters
        ----------
        subsample_n : int
            The number of signal traces to randomly sample from the full dataset
        weighted : bool
            Whether to weight the likelihood of rows being selected to the random
            sample based on the total number of each label in the dataset

        Returns
        -------
        pd.DataFrame containing metadata for the subsample of signal traces'''
        print('Fetching subsample of traces from hdf5 files')
        if self.weighted:
            print('Weighting random sample based on category label')
            label_counts = self.full_metadata['trace_category'].value_counts()
            self.full_metadata['weight_for_subsample'] = 1./self.full_metadata.groupby('trace_category')['trace_category'].transform('count')
            subsample_metadata = self.full_metadata.sample(self.subsample_n, weights=self.full_metadata['weight_for_subsample'], random_state=0)
        else:
            subsample_metadata = self.full_metadata.sample(self.subsample_n, random_state=0)
        subsample_metadata.set_index('trace_name', drop=True, inplace=True)
        return subsample_metadata

    def _read_h5py_files(self, trace_names):
        '''Reads seismic signal trace data for the indicated trace names
        from the h5py files in the directory specified in the class init

        Parameters
        ----------
        trace_names : list of str
            A list of trace names to fetch data for from the .h5py files

        Returns
        -------
        dict : A dictionary where the keys are the trace name strings, and the values
        are the corresponding seismic signal trace data arrays (np.array format)'''
        traces = {}
        print(f'Parsing traces from h5py filepaths')
        for i, path in enumerate(tqdm(self.data_paths)):
            h5f = h5py.File(path, mode='r')
            for _, trace_name in enumerate(trace_names):
                try:
                    trace = np.array(h5f.get(f'data/{trace_name}'))
                    if self._is_populated(trace):
                        traces[trace_name] = trace
                except Exception as e:
                    pass
        return traces


    def create_waveform_images(self, img_width=6, img_height=2, img_dpi=100):
        '''Iterates through the signal traces, plotting a waveform image for each.
        Saves each created image to an array to be used for model training.

        Parameters
        ----------
        img_width : int
            The width of the waveform image to plot, in inches
        img_height : int
            The height of the waveform image to plot, in inches
        img_dpi : int
            The resolution to save the image at (dots per inch)

        Returns
        -------
        Array of images, where each element in the array is a waveform image of shape
        (img_height, img_width, 3). The 3 corresponds to the 3 color channels RGB.'''

        print('Creating waveform images from signal traces')
        self.subsample_waveform_imgs = np.zeros(
            (len(self.subsample_traces.keys()), img_height*img_dpi, img_width*img_dpi, 3),
            dtype=np.uint8
        )

        i = 0
        for trace_name, trace in tqdm(self.subsample_traces.items()):
            try:
                # Convert the signal to a spectrogram image
                img_arr = plot_waveform(
                    trace[:,2], # select only the z-axis component of the signal
                    img_width=img_width,
                    img_height=img_height,
                    dpi=img_dpi
                    )
                self.subsample_waveform_imgs[i] = img_arr
                del img_arr
            except Exception as e:
                print(
                    f'Unable to plot waveform for {trace_name}.'
                    f'Saving empty array for this trace. \n Exception: {e}'
                )
            if i % 1000 == 0:
                gc.collect() # garbage collection of deleted objects to free memory
            i += 1

        return self.subsample_waveform_imgs


    def create_spectrogram_images(self, img_width=3, img_height=2, img_dpi=100):
        '''Iterates through the signal traces, plotting a spectrogram image for each.
        Saves each created image to an array to be used for model training.

        Parameters
        ----------
        img_width : int
            The width of the waveform image to plot, in inches
        img_height : int
            The height of the waveform image to plot, in inches
        img_dpi : int
            The resolution to save the image at (dots per inch)

        Returns
        -------
        Array of images, where each element in the array is a spectrogram image of shape
        (img_height, img_width, 3). The 3 corresponds to the 3 color channels RGB.'''

        print('Creating spectrogram images from signal traces')
        self.subsample_spectrogram_imgs = np.zeros(
            (len(self.subsample_traces.keys()), img_height*img_dpi, img_width*img_dpi, 3),
            dtype=np.uint8
        )

        i = 0
        for trace_name, trace in tqdm(self.subsample_traces.items()):
            try:
                # Convert the signal to a spectrogram image
                img_arr = plot_spectrogram(
                    trace[:,2], # select only the z-axis component of the signal
                    img_width=img_width,
                    img_height=img_height,
                    dpi=100
                )
                self.subsample_spectrogram_imgs[i] = img_arr
                del img_arr

            except Exception as e:
                print(
                    f'Unable to plot spectrogram for {trace_name}.'
                    f'Saving empty array for this trace. \n Exception: {e}'
                )
            if i % 1000 == 0:
                gc.collect() # garbage collection of deleted objects to free memory

            i += 1

        return self.subsample_spectrogram_imgs
