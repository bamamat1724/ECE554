

import os
import glob
import time
import soundfile as sf
import librosa
import librosa.display
import subprocess


def play_audio(filename=None, data=None, sample_rate=None):
    if filename:
        print("Play audio:", filename)
        subprocess.call(["cvlc", "--play-and-exit", filename])
    else:
        print("Play audio data")
        filename = '.tmp_audio_from_play_audio.wav'
        write_audio(filename, data, sample_rate)
        subprocess.call(["cvlc", "--play-and-exit", filename])


def read_audio(filename, dst_sample_rate=16000, PRINT=False):
    if 0:  # This takes 0.4 seconds to read an audio of 1 second. But support for more format
        data, sample_rate = librosa.load(filename)
    else:  # This only takes 0.01 seconds
        data, sample_rate = sf.read(filename)

    assert len(data.shape) == 1, "This project only support 1 dim audio."

    if (dst_sample_rate is not None) and (dst_sample_rate != sample_rate):
        data = librosa.core.resample(data, orig_sr=sample_rate, target_sr=dst_sample_rate)
        sample_rate = dst_sample_rate

    if PRINT:
        print("Read audio file: {}.\n Audio len = {:.2}s, sample rate = {}, num points = {}".format(
            filename, data.size / sample_rate, sample_rate, data.size))
    return data, sample_rate


def write_audio(filename, data, sample_rate, dst_sample_rate=16000):
    if (dst_sample_rate is not None) and (dst_sample_rate != sample_rate):
        data = librosa.core.resample(data, orig_sr=sample_rate, target_sr=dst_sample_rate)
        sample_rate = dst_sample_rate

    sf.write(filename, data, sample_rate)
    # librosa.output.write_wav(filename, data, sample_rate)


def write_list(filename, data):
    with open(filename, 'w') as f:
        for d in data:
            f.write(str(d) + "\n")
        # What's in file: "[2, 3, 5]\n[7, 11, 13, 15]\n"


def read_list(filename):
    with open(filename) as f:
        with open(filename, 'r') as f:
            data = [l.rstrip() for l in f.readlines()]
    return data


def create_folder(folder):
    print("Creating folder:", folder)
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def files(folder, file_types=('*.wav',)):
    filenames = []
    
    if not isinstance(file_types, tuple):
        file_types = [file_types]
        
    for file_type in file_types:
        filenames.extend(glob.glob(folder + "/" + file_type))
    filenames.sort()
    return filenames

def get_dir_names(folder):
    names = [name for name in os.listdir(folder) if os.path.isdir(name)] 
    return names 

def get_all_names(folder):
    return os.listdir(folder)

def change_suffix(s, new_suffix, index=None):
    i = s.rindex('.')
    si = ""
    if index:
        si = "_" + str(index)
    s = s[:i] + si + "." + new_suffix
    return s 

def int2str(num, len):
    return ("{:0"+str(len)+"d}").format(num)

def add_idx_suffix(s, idx): # /data/two.wav -> /data/two_032.wav
    i = s.rindex('.')
    s = s[:i] + "_" + "{:03d}".format(idx) + s[i:]
    return s

class Timer(object):
    def __init__(self):
        self.t0 = time.time()
    def report_time(self, event="", prefix=""):
        print(prefix + "Time cost of '{}' is: {:.2f} seconds.".format(
            event, time.time() - self.t0
        ))
