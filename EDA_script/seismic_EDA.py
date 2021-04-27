'''
This script requires that you have made a directory of images using the "create_images.py" script contained in this repository first.

This script imports earthquake signal data and metadata for files in the created images directory and uses it to plot the following plots:
    1. Plot of a single example waveform/spectrogram/PSD for an earthquake
    2. Plot of distributions of earthquake magnitude, depth, and source-receiver distance for all earthquakes in the image dataset
    3. Plot of global earthquake locations in the image dataset
    4. Plot of global seismometer locations in the image dataset
    
Please enter user input from lines 43-60. User input requires you to define filepaths to signal data, metadata, and image data as shown below.

Created by Kaelynn Rose
on 3/31/2021

'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py

from skimage import io, color, filters
from skimage.transform import resize, rotate
from skimage.color import rgb2gray
import cv2

import tensorflow as tf
from tensorflow.keras import layers
keras = tf.keras

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

import cartopy
from cartopy import config
import cartopy.crs as ccrs

plt.style.use('ggplot')

###################### USER INPUT ####################

# paths to csv and hdf5 (waveform/signal) files
noise_csv_path = 'data/chunk1/chunk1.csv'
noise_sig_path = 'data/chunk1/chunk1.hdf5'
eq1_csv_path = 'data/chunk2/chunk2.csv'
eq1_sig_path = 'data/chunk2/chunk2.hdf5'
eq2_csv_path = 'data/chunk3/chunk3.csv'
eq2_sig_path = 'data/chunk3/chunk3.hdf5'
eq3_csv_path = 'data/chunk4/chunk4.csv'
eq3_sig_path = 'data/chunk4/chunk4.hdf5'
eq4_csv_path = 'data/chunk5/chunk5.csv'
eq4_sig_path = 'data/chunk5/chunk5.hdf5'
eq5_csv_path = 'data/chunk6/chunk6.csv'
eq5_sig_path = 'data/chunk6/chunk6.hdf5'

# directory to pull images from
dir = 'images/big_data_random/specs'

###################### END USER INPUT ####################

### Organize data

# read the noise and earthquake csv files into separate dataframes:
earthquakes_1 = pd.read_csv(eq1_csv_path)
earthquakes_2 = pd.read_csv(eq2_csv_path)
earthquakes_3 = pd.read_csv(eq3_csv_path)
earthquakes_4 = pd.read_csv(eq4_csv_path)
earthquakes_5 = pd.read_csv(eq5_csv_path)
noise = pd.read_csv(noise_csv_path)

# create a csv which combines all of the dataframes
full_csv = pd.concat([earthquakes_1,earthquakes_2,earthquakes_3,earthquakes_4,earthquakes_5,noise])
print(f'csv file length: {len(full_csv)}')

# filtering the dataframe: uncomment if needed
#df = df[(df.trace_category == 'earthquake_local') & (df.source_distance_km <= 20) & (df.source_magnitude > 3)]
#print(f'total events selected: {len(df)}')

# making lists of trace names for the individual signal datasets
eq1_list = earthquakes_1['trace_name'].to_list()
eq2_list = earthquakes_2['trace_name'].to_list()
eq3_list = earthquakes_3['trace_name'].to_list()
eq4_list = earthquakes_4['trace_name'].to_list()
eq5_list = earthquakes_5['trace_name'].to_list()
noise_list = noise['trace_name'].to_list()

### Get traces names for all signals in the images dataset
traces_array = []
for filename in os.listdir(dir):
    if filename.endswith('.png'):
        traces_array.append(filename[0:-4])
        
# select only the rows in the metadata dataframe which correspond to images
img_dataset = full_csv.loc[full_csv['trace_name'].isin(traces_array)]
labels = img_dataset['trace_category'] # target variable, 'earthquake' or 'noise'
labels = labels.map(lambda x: 1 if x == 'earthquake_local' else 0) # transform target variable to numerical categories
labels = np.array(labels)
len(img_dataset)


# FIGURE FUNCTIONS

### Figure 1: Earthquake Source Locations Global Plot

def waveform_spectrogram_plot(signal_path,signal_index,signal_list):
    dtfl = h5py.File(signal_path, 'r') # find the signal file
    dataset = dtfl.get('data/'+str(signal_list[signal_index])) # fetch one signal from the file
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data = np.array(dataset)

    # plot
    fig, (ax1,ax2,ax3) = plt.subplots(3,1,figsize=(9,7))
    ax1.plot(np.linspace(0,60,6000),data[:,2],color='k',linewidth=1) # plot waveform
    ymin, ymax = ax1.get_ylim()
    ax1.vlines(dataset.attrs['p_arrival_sample']/100,ymin,ymax,color='b',linewidth=1.5, label='P-arrival') # plot p-wave arrival time
    ax1.vlines(dataset.attrs['s_arrival_sample']/100, ymin, ymax, color='r', linewidth=1.5, label='S-arrival') # plot s-wave arrival time
    ax1.vlines(dataset.attrs['coda_end_sample']/100, ymin, ymax, color='cyan', linewidth=1.5, label='Coda end')
    ax1.set_xlim([0,60])
    ax1.legend(loc='lower right',fontsize=10)
    ax1.set_ylabel('Amplitude (counts)')
    ax1.set_xlabel('Time (s)')
    im = ax2.specgram(data[:,2],Fs=100,NFFT=256,cmap='jet',vmin=-10,vmax=25); # plot spectrogram
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('Time (s)')
    ax3.psd(data[:,2],256,100,color='cornflowerblue') # plot power spectral density
    ax3.set_xlim([0,50])
    plt.savefig('waveform_spectrogram_plot.png',dpi=500)
    plt.tight_layout()
    plt.show()

    print('The p-wave for this waveform was picked by: ' + dataset.attrs['p_status'])
    print('The s-wave for this waveform was picked by: ' + dataset.attrs['s_status'])

# test the function (will produce a plot of the waveform, spectrogram, and power-spectral-density)
waveform_spectrogram_plot(eq1_sig_path,8020,eq1_list)


### Figure 2: PLot of Earthquake magnitude, depth and distance

def plot_mags_depths_distance():
    depths = img_dataset['source_depth_km'].dropna() # clean depth data
    depths = np.array(depths)
    depths = [float(x) for x in depths if x != 'None'] # get the earthquake source depths

    # plot
    plt.style.use('seaborn-pastel')
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(16,5))
    ax1.hist(img_dataset['source_magnitude'],bins=100) # plot earthquake source magnitude
    ax1.set_xlabel('Earthquake Magnitude',fontsize=14)
    ax1.set_ylabel('Frequency',fontsize=14)
    ax1.set_title('Earthquake Magnitude',fontsize=18)
    ax2.hist(depths, bins=300)
    ax2.set_xlabel('Earthquake Source Depth (km)',fontsize=14) # plot earthquake source depth
    ax2.set_ylabel('Frequency',fontsize=14)
    ax2.set_xlim([-10, 100])
    ax2.set_title('Earthquake Depth',fontsize=18)
    ax3.hist(img_dataset['source_distance_km'],bins=100)
    ax3.set_xlabel('Source-Receiver Distance (km)',fontsize=14) # plot distance from earthquake to station
    ax3.set_ylabel('Frequency',fontsize=14)
    ax3.set_title('Earthquake Distance',fontsize=18)
    ax3.set_xlim([-10, 250])
    plt.savefig('mags_depths_distances.png',dpi=500)
    plt.tight_layout()
    plt.show()

# test the plotting function
plot_mags_depths_distance() # will produce a plot of eathquake magnitudes, depths, and distances in the full image dataset


### Figure 3: Plot of earthquake locations

def plot_eq_locations(dotcolor,dotsize,dotshape):
    lats = img_dataset['source_latitude'] # get earthquake latitudes
    lons = img_dataset['source_longitude'] # get earthquake longitudes
    sizes = img_dataset['source_magnitude'] # get earthquake magnitudes to plot point sizes

    fig, ax = plt.subplots(figsize=(18,13))
    ax = plt.axes(projection=ccrs.PlateCarree()) # choose a map projection using Cartopy
    ax.coastlines() # add coastlines to map
    ax.add_feature(cartopy.feature.OCEAN,color='lightskyblue',alpha=0.3) # fill ocean with color
    ax.add_feature(cartopy.feature.LAND, facecolor='gainsboro',edgecolor='black',alpha=0.3) # fill land with color
    ax.gridlines(color='white')
    ax.scatter(lons,lats,c='red',s=10, marker='o', transform=ccrs.PlateCarree(),label='earthquakes') # add points represnting each earthquake location
    ax.set_xlim([-360,360])
    ax.set_ylim([-90,90])
    ax.legend()
    plt.savefig('earthquake_map.png',dpi=500)
    plt.show()

# test the plotting function
plot_eq_locations('red',10,'o') # plot marker color, size, shape


### Figure 4: Plot of seismometer locations
# Make a worldmap plot of earthquake locations by magnitude and depth

def plot_stations(dotcolor,dotsize,dotshape):

    lats = img_dataset['receiver_latitude'] # get seismic station latitudes
    lons = img_dataset['receiver_longitude'] # get seismic station longitudes

    # plot
    fig, ax = plt.subplots(figsize=(18,13))
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.add_feature(cartopy.feature.OCEAN,color='lightskyblue',alpha=0.3)
    ax.add_feature(cartopy.feature.LAND, facecolor='gainsboro',edgecolor='black',alpha=0.3)
    ax.gridlines(color='white')
    ax.scatter(lons,lats,c='blue',marker='v',s=20,transform=ccrs.PlateCarree(),label='stations')
    ax.set_xlim([-360,360])
    ax.set_ylim([-90,90])
    ax.legend()
    plt.savefig('station_map.png',dpi=500)
    plt.show()

# test the plotting function
plot_stations('blue',20,'v') # plot marker color, size, shape


### Figure 5: example plot of earthquake signals vs. noise signals being used to train the CNN

def eq_vs_noise_plot():
    dtfl = h5py.File(eq1_sig_path, 'r')
    dataset1 = dtfl.get('data/'+str(eq1_list[8021])) # select a random trace name from the list
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data1 = np.array(dataset1)

    dtf2 = h5py.File(eq2_sig_path, 'r')
    dataset2 = dtf2.get('data/'+str(eq2_list[8020])) # select a random trace name from the list
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data2 = np.array(dataset2)

    dtf3 = h5py.File(eq3_sig_path, 'r')
    dataset3 = dtf3.get('data/'+str(eq3_list[155555])) # select a random trace name from the list
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data3 = np.array(dataset3)

    dtf4 = h5py.File(noise_sig_path, 'r')
    dataset4 = dtf4.get('data/'+str(noise_list[6524])) # select a random trace name from the list
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data4 = np.array(dataset4)

    dtf5 = h5py.File(noise_sig_path, 'r')
    dataset5 = dtf5.get('data/'+str(noise_list[8111])) # select a random trace name from the list
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data5 = np.array(dataset5)

    dtf6 = h5py.File(noise_sig_path, 'r')
    dataset6 = dtf6.get('data/'+str(noise_list[9333])) # select a random trace name from the list
    # waveforms, 3 channels: first row: E channel, second row: N channel, third row: Z channel
    data6 = np.array(dataset6)

    fig, axs = plt.subplots(2,3,figsize=(12,8))
    datasets = [dataset1,dataset2,dataset3,dataset4,dataset5,dataset6]
    datas = [data1,data2,data3,data4,data5,data6]
    titles = ['earthquake','earthquake','earthquake','noise','noise','noise']
    for i, ax in enumerate(axs.flatten()):
        im = ax.specgram(datas[i][:,2],Fs=100,NFFT=256,cmap='gray',vmin=-10,vmax=25)
        ax.set_ylabel('Frequency (Hz)',fontsize=12)
        ax.set_xlabel('Time (s)',fontsize=12)
        ax.set_title(titles[i],fontsize=14)

    plt.tight_layout()
    plt.savefig('earthquakes_vs_noise_cnn_images.png')
    plt.show()

# use this plotting function
eq_vs_noise_plot()

