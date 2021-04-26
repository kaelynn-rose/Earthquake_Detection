'''



Created by Kaelynn Rose
on 4/22/2021

'''

import numpy as np
import matplotlib.pyplot as plt
import boto3
import obspy
from obspy.clients.seedlink.easyseedlink import create_client

traces = []

def handle_data(trace):
    global traces
    print('Received the following trace:')
    print(trace)
    print()
    traces.append(trace)
    
    if len(traces) > 9:
        data_packet = traces[0:9]
        traces = traces[5:]
        print('Shortened trace')
        data_packet = data_packet[0] + data_packet[1] + data_packet[2] + data_packet[3] + data_packet[4] + data_packet[5] + data_packet[6] + data_packet[7] + data_packet[8]
        print(f'The data packet is {data_packet}')

        fig, ax = plt.subplots(figsize=(3,2))
        ax.specgram(data_packet.data,Fs=100,NFFT=256,cmap='gray',vmin=-10,vmax=20);
        ax.set_xlim([0,60])
        ax.axis('off')
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, hspace = 0, wspace = 0)
        plt.margins(0,0)
        filename = str(data_packet)[0:2]+'_'+str(data_packet)[3:7]+'_'+str(data_packet)[9:12]+'_'+str(data_packet)[15:28]+'_'+str(data_packet)[29:31]+'_'+str(data_packet)[32:34]+'_'+str(data_packet)[35:42]+'.png'
        plt.savefig(filename,bbox_inches='tight',transparent = True,pad_inches=0,dpi=50)
        plt.close()
        s3 = boto3.resource('s3')
        s3.meta.client.upload_file(filename, 'seismic-input-lambda', filename)
    
    
    
client = create_client('rtserve.iris.washington.edu', on_data=handle_data)

# Send the INFO:ID request
client.get_info('ID')

# Returns:
# <?xml version="1.0"?>\n<seedlink software="SeedLink v3.2 (2014.071)" organization="GEOFON" started="2014/09/01 14:08:37.4192"/>\n

client.capabilities
#['dialup', 'multistation', 'window-extraction', 'info:id', 'info:capabilities', 'info:stations', 'info:streams']

#client.get_info('STREAMS')
client.select_stream('HV', 'AHUD', 'EHZ')

client.run()
