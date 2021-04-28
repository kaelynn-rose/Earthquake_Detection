'''
This script uses the ObsPy toolbox to stream data in near-real time from a seismic station on Kilauea volcano, Hawaii, via the Incorporated Research Institutions for Seismology (IRIS) Seedlink client. Data is fetched every ~30 seconds. As the signal data is streamed, the script generates a spectrogram figure every 20 seconds, in the same format as the spectrograms used to train/test the CNN classification model (seismic_CNN.py) found in this repo.

Process:
1. Get signal data streamed from IRIS
2. Create a spectrogram figure every 20 seconds
3. Save the spectrogram figure to the specified AWS s3 bucket
4. Saving any object to this s3 bucket triggers the lambda function
5. The AWS lambda function (found in the lambda-earthquake-cnn folder in this repo) predicts the class of each figure using the CNN model (results appear in CloudWatch)

Created by Kaelynn Rose
on 4/22/2021

'''

import numpy as np
import matplotlib.pyplot as plt
import boto3
import obspy
from obspy.clients.seedlink.easyseedlink import create_client

traces = [] # initialize blank list for traces

# function to fetch traces, create figure for each trace, and save trace to AWS s3 bucket
def handle_data(trace):

    global traces
    print('Received the following trace:')
    print(trace)
    print()
    traces.append(trace)
    
    # if more than 9 traces in traces variable, create a spectrogram figure
    if len(traces) > 9:
        data_packet = traces[0:9] # data to plot (~60 seconds)
        traces = traces[2:] # remove the first 2 traces from the traces variable
        print('Shortened trace')
        data_packet = data_packet[0] + data_packet[1] + data_packet[2] + data_packet[3] + data_packet[4] + data_packet[5] + data_packet[6] + data_packet[7] + data_packet[8] # merge signal traces into one signal to plot
        print(f'The data packet is {data_packet}')

        # plot spectrogram in the same way as the spectrograms used to train the CNN classification model (seismic_CNN.py)
        fig, ax = plt.subplots(figsize=(3,2))
        ax.specgram(data_packet.data,Fs=100,NFFT=256,cmap='gray',vmin=-10,vmax=20); # plot spectrogram
        ax.set_xlim([0,60])
        ax.axis('off')
        plt.gca().set_axis_off() # remove axes
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        filename = str(data_packet)[0:2]+'_'+str(data_packet)[3:7]+'_'+str(data_packet)[9:12]+'_'+str(data_packet)[15:28]+'_'+str(data_packet)[29:31]+'_'+str(data_packet)[32:34]+'_'+str(data_packet)[35:42]+'.png'
        plt.savefig(filename,bbox_inches='tight',transparent = True,pad_inches=0,dpi=50) # save figure
        plt.close()
        s3 = boto3.resource('s3')
        s3.meta.client.upload_file(filename, 'seismic-input-lambda', filename) # upload spectrogram to s3 bucket
    
    
# specify client (IRIS) to retrieve data from
client = create_client('rtserve.iris.washington.edu', on_data=handle_data)

# Send the INFO:ID request
client.get_info('ID')

# print capabilities
client.capabilities

#client.get_info('STREAMS')

# specify stream
client.select_stream('HV', 'AHUD', 'EHZ') # network, station, channel

# run the client, this will begin streaming the data
client.run()
