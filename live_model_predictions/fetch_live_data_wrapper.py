"""This script is a wrapper for fetch_live_data.py, to restart the script in
the event of a timeout from the Obspy EasySeedLink client."""

import subprocess
import time

MAX_RETRIES = 5  # Maximum number of retries

def run_script():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            # Set command to run
            command = [
                'python3', 'fetch_live_data.py',
                '-network', 'HV',
                '-station', 'AHUD',
                '-channel', 'EHZ',
                '-bucket', 'earthquake-detection'
            ]
            # Run the script as a subprocess
            result = subprocess.run(
                command,
                check=True, # Raise an exception if the script fails
            )
            print(f'Script running successfully with return code {result.returncode}')
        except Exception as e:
            print(f'Script failed with error: {e}. Restarting...')
        retries += 1
        print(f'Retrying... {retries}/{MAX_RETRIES}')
        time.sleep(5)  # Seconds to wait before restarting

if __name__ == "__main__":
    run_script()