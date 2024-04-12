import sys, os
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
sys.path.append(ROOT)

import common.record_audio as rec
import time

# Set up audio recorder
recorder = rec.AudioRecorder()

# start recording
recorder.start_record(folder='./data/data_tmp/')

# Do anything here
# time.sleep(5)

# stop recording
recorder.stop_record()

