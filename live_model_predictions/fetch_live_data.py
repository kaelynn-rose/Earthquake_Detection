"""
This script fetches live data from a specified seismic network/station/channel
using an ObsPy client to connect to the SeedLink server. The data handling function
saves the seismic trace data in 60 second intervals with a 10 second sliding window.
So the first saved data packet will be 0-60s, the second will be 10-70s,
the third will be 20-80s, etc. The script saves each data packet to a /live_data
directory within the specified S3 bucket. This script uses plac annotations to pass the
desired parameters into the function; see Example Usage for how to run this script
from the command line.

Example Usage
-------------
# To get a live stream of data from the Hawaiian Volcano Observatory (HVO)
# network, station AHUD, channel EHZ (Extremely short period High gain Z-axis
# seismometer that records motion in the up-down direction), and save it to the
# /live_data directory in the S3 bucket named 'earthquake-detection'

>> python3 fetch_live_data.py -network 'HV' -station 'AHUD' -channel 'EHZ' -bucket 'earthquake-detection'
"""

import boto3
import numpy as np
import plac

from obspy.clients.seedlink.easyseedlink import create_client


def handle_data(trace):
    """A function that handles new data received from the SeedLink server, and
    saves a snapshot of the live data  array to an S3 bucket in .npy format.

    Parameters
    ----------
    trace : obspy.core.trace
        An ObsPy trace object that contains live stream data and metadata"""
    # Access global variables that are needed for data handling
    global bucket
    global traces
    if traces == []:
        traces = trace
    else:
        traces += trace

    # Add trace to trace stream
    print(f'Received the following trace:\n {trace}')
    print(f'The trace stream now contains {traces.stats.npts} points')

    if traces.stats.npts >= 6000:
        starttime = traces.stats.starttime
        snapshot = traces.slice(starttime=starttime, endtime=starttime+60.0) # get 60s snapshot of the signal
        traces = traces.trim(starttime=starttime+10.0) # trim the first 6s from the signal to slide the window forward
        print(f'Taking signal snapshot: {snapshot}')

        # Get trace metadata for file naming
        network = snapshot.stats['network']
        station = snapshot.stats['station']
        channel = snapshot.stats['channel']
        sampling_rate = snapshot.stats['sampling_rate']
        starttime = snapshot.stats['starttime']
        endtime = snapshot.stats['endtime']

        # Save snapshot data packet to .npy and upload to S3
        print('Saving signal snapshot to S3')
        s3 = boto3.resource('s3')
        file_path = (
            f'live_data/{network}_{station}_{channel}_{sampling_rate}Hz_{starttime}_{endtime}.npy'
        )
        np.save('tmp.npy', snapshot)
        s3.meta.client.upload_file('tmp.npy', bucket, file_path) # upload array file to S3
    print('\n')

@plac.annotations(
    network=('Seismic network name', 'option', 'network', str),
    station=('Station name', 'option', 'station', str),
    channel=('Channel name', 'option', 'channel', str),
    s3_bucket=('S3 bucket name', 'option', 'bucket', str)
)
def main(network, station, channel, s3_bucket):
    """Main function used to create and run an ObsPy client that connects to
    a SeedLink server to fetch a live stream of monitoring data. See plac function
    call for parameter details."""
    # Globalize variables so they can be used within data handling function
    global bucket
    bucket = s3_bucket
    global traces
    traces = []

    # Create and run an ObsPy client to connect to a SeedLink server
    client = create_client('rtserve.iris.washington.edu', on_data=handle_data)
    client.select_stream(network, station, channel)
    client.run()


if __name__ == '__main__':
    plac.call(main)