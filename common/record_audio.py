#!/usr/bin/env python3


import time
from threading import Thread
import subprocess
import os
import sounddevice as sd
import soundfile as sf
import argparse, tempfile, queue, sys, datetime


class AudioRecorder(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser(description=__doc__)
        self.parser.add_argument(
            '-l', '--list-devices', action='store_true',
            help='show list of audio devices and exit')
        self.parser.add_argument(
            '-d', '--device', type=int, default='0',
            help='input device (numeric ID or substring)')
        self.parser.add_argument(
            '-r', '--samplerate', type=int, help='sampling rate')
        self.parser.add_argument(
            '-c', '--channels', type=int, default=1, help='number of input channels')
        self.parser.add_argument(
            'filename', nargs='?', metavar='FILENAME',
            help='audio file to store recording to')
        self.parser.add_argument(
            '-t', '--subtype', type=str, help='sound file subtype (e.g. "PCM_24")')

        self.args = self.parser.parse_args()

        if self.args.list_devices:
            print(sd.query_devices())
            self.parser.exit(0)
        if self.args.samplerate is None:
            device_info = sd.query_devices(self.args.device, 'input')
            # soundfile expects an int, sounddevice provides a float:
            self.args.samplerate = int(device_info['default_samplerate'])

    def start_record(self, folder='./', micThreshold=0.1):

        if not os.path.exists(folder):
            os.makedirs(folder)

        self.thread_record = Thread(target = self.record, args=(folder, micThreshold, ))
        self.thread_record.start()

    def stop_record(self):
        self.thread_record.join()

        time_duration = time.time() - self.audio_time0
        print('File saved! ' + self.filename)

    def record(self, folder, volThresh):
        q = queue.Queue()

        def callback(indata, frames, time, status):
            #This is called (from a separate thread) for each audio block.
            if status:
                print(status, file=sys.stderr)
            new_val = indata.copy()
            # print(new_val)
            q.put(new_val)

        self.filename = tempfile.mktemp(
            prefix= 'audio_' + self.get_time(),
            suffix='.wav',
            dir=folder)
        self.audio_time0 = time.time()
        with sf.SoundFile(self.filename, mode='x', samplerate=self.args.samplerate,
                        channels=self.args.channels, subtype=self.args.subtype) as file:
            with sd.InputStream(samplerate=self.args.samplerate, device=self.args.device,
                                channels=self.args.channels, callback=callback):

                q.get() # Flush data

                print('#' * 80)
                print('Start recording:')
                print('#' * 80)
                # while True and self._thread_alive:
                talkFlag = False
                offTime = 3
                timeout = 10

                while True:
                    data = q.get() # Get the data
                    if self.isTalking(data, volThresh):
                        if not talkFlag: # IF first time talking
                            onStart = time.time() # Update start time
                            talkFlag = True  # update flag
                        offStart = time.time() # Always update off time timeout
                        if time.time()-onStart > timeout: #timeout if mic is on for long time
                            break
                    else:
                        if talkFlag and time.time() - offStart > offTime: #timeout if no talking for long enough
                            talkFlag = False
                            break
                    if talkFlag:
                        file.write(data)

    def isTalking(self, data, threshold):
        return max(abs(data)) > threshold

    def get_time(self):
        s=str(datetime.datetime.now())[5:].replace(' ','-').replace(":",'-').replace('.','-')[:-3]
        return s # day, hour, seconds: 02-26-15-51-12-556


if __name__ == '__main__':

    # Set up audio recorder
    recorder = AudioRecorder()

    # start recording
    recorder.start_record(folder='./data/data_tmp/')

    time.sleep(5)

    # stop recording
    recorder.stop_record()
