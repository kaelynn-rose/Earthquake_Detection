'''



Created by Kaelynn Rose
on 4/22/2021

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

import tensorflow as tf
from tensorflow.keras import layers
keras = tf.keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from datetime import datetime
from joblib import Parallel,delayed

from scipy import signal
from scipy.signal import resample,hilbert

############################# USER INPUT #############################

datapath = '/Users/kaelynnrose/Saved_Documents/Documents/GALVANIZE/Capstones/Capstone_2/'

# paths to csv and hdf5 (waveform/signal) files
noise_csv_path = datapath+'data/chunk1/chunk1.csv'
noise_sig_path = datapath+'data/chunk1/chunk1.hdf5'
eq1_csv_path = datapath+'data/chunk2/chunk2.csv'
eq1_sig_path = datapath+'data/chunk2/chunk2.hdf5'
eq2_csv_path = datapath+'data/chunk3/chunk3.csv'
eq2_sig_path = datapath+'data/chunk3/chunk3.hdf5'
eq3_csv_path = datapath+'data/chunk4/chunk4.csv'
eq3_sig_path = datapath+'data/chunk4/chunk4.hdf5'
eq4_csv_path = datapath+'data/chunk5/chunk5.csv'
eq4_sig_path = datapath+'data/chunk5/chunk5.hdf5'
eq5_csv_path = datapath+'data/chunk6/chunk6.csv'
eq5_sig_path = datapath+'data/chunk6/chunk6.hdf5'

traces_path = '/Users/kaelynnrose/Saved_Documents/Documents/GALVANIZE/Capstones/Capstone_2/images/big_data_random/waves_long'

num_traces = 100000

#######################################################################


# read the noise and earthquake csv files into separate dataframes:
noise = pd.read_csv(noise_csv_path)
earthquakes_1 = pd.read_csv(eq1_csv_path)
earthquakes_2 = pd.read_csv(eq2_csv_path)
earthquakes_3 = pd.read_csv(eq3_csv_path)
earthquakes_4 = pd.read_csv(eq4_csv_path)
earthquakes_5 = pd.read_csv(eq5_csv_path)

# combine all csv files into one
full_csv = pd.concat([earthquakes_1,earthquakes_2,earthquakes_3,earthquakes_4,earthquakes_5,noise])

# get a list of all the traces in selected folder
traces_array = []
for filename in os.listdir(traces_path): # loop through every file in the directory and get trace names from the image files
    if filename.endswith('.png'):
        traces_array.append(filename[0:-4]) # remove .png from image file names



def get_traces(num_traces):

    partial_dataset = [] # initialize dataset
    partial_df = [] # initialize dataframe
    counter = 0
    random_traces = np.random.choice(traces_array,num_traces,replace=False) # get x number of random traces

    for i, trace in enumerate(random_traces): # loop through traces to get signal data
        counter += 1
        print(f'Working on trace # {counter}')
        
        csv_row = full_csv.loc[full_csv['trace_name'] == trace] # find corresponding row of metadata
        array_row = csv_row.iloc[0].to_numpy()
        partial_df.append(array_row) # append metadata to new dataframe
        
        if trace in list(earthquakes_1['trace_name']):
            dtfl = h5py.File(eq1_sig_path, 'r') # access signal file
            data = np.array(dtfl.get('data/'+str(trace))) # get data for that specific trace
            partial_dataset.append(data) # append signal data row to dataset
            print(trace)
        elif trace in list(earthquakes_2['trace_name']):
            dtfl = h5py.File(eq2_sig_path, 'r') # access signal file
            data = np.array(dtfl.get('data/'+str(trace))) # get data for that specific trace
            partial_dataset.append(data) # append signal data row to dataset
            print(trace)
        elif trace in list(earthquakes_3['trace_name']):
            dtfl = h5py.File(eq3_sig_path, 'r') # access signal file
            data = np.array(dtfl.get('data/'+str(trace))) # get data for that specific trace
            partial_dataset.append(data) # append signal data row to dataset
            print(trace)
        elif trace in list(earthquakes_4['trace_name']):
            dtfl = h5py.File(eq4_sig_path, 'r') # access signal file
            data = np.array(dtfl.get('data/'+str(trace))) # get data for that specific trace
            partial_dataset.append(data) # append signal data row to dataset
            print(trace)
        elif trace in list(earthquakes_5['trace_name']):
            dtfl = h5py.File(eq5_sig_path, 'r') # access signal file
            data = np.array(dtfl.get('data/'+str(trace))) # get data for that specific trace
            partial_dataset.append(data) # append signal data row to dataset
            print(trace)
        elif trace in list(noise['trace_name']):
            dtfl = h5py.File(noise_sig_path, 'r') # access signal file
            data = np.array(dtfl.get('data/'+str(trace))) # get data for that specific trace
            partial_dataset.append(data) # append signal data row to dataset
            print(trace)
        else:
            print('Trace not found in dataframes')
    
    filtered_df = pd.DataFrame(partial_df,columns=list(full_csv.columns)) # convert final array to dataframe
    np.save('partial_signal_dataset'+str(num_traces)+'.npy',partial_dataset) # save signal dataset
    filtered_df.to_csv('partial_signal_df'+str(num_traces)+'.csv') # save dataframe to csv

# use the get_traces function
get_traces(100000) # fetch waveform data and labels for 100000 traces






