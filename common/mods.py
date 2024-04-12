import numpy as np
import librosa
import sys, os

# Prepare to add custom libraries
ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
sys.path.append(ROOT)

# Import supporting files
import common.util as util

def matchDataToLen(noise, dataLen):
    if len(noise) < dataLen:
        noise = np.tile(noise, 1+(dataLen//len(noise)))
        nStart = 0
    else:
        nStart = np.random.randint(len(noise) - dataLen + 1)
    return noise[nStart:nStart + dataLen]


def clipData(data, limit):
    data[data > limit] = limit
    data[data < -limit] = -limit
    return data


class Noiser(object):
    def __init__(self, mods, chance=1.00):
        self.mods = mods
        self.chance = chance
        
    def __call__(self, audio):
        if np.random.random() > self.chance:
            return audio
        else:
            for mod in self.mods:
                audio = mod(audio)
            return audio

    class recordedNoise(object):
        def __init__(self, dir, prob_noise=0.5, amplitude=(0, 0.5)):
            self.amplitude = amplitude

            fnames = util.files(dir)
            noises = []
            for name in fnames:
                noise, rate = util.read_audio(filename=name)
                noise = librosa.util.normalize(noise)
                noise = self.padNoise(noise, rate, time=10)
                noises.append(noise)
            self.noises = noises
            self.prob_noise = prob_noise
            
        def __call__(self, audio):
            if np.random.random() > self.prob_noise:
                return audio
            
            data = audio.data
            
            # add noise
            noise = self.selNoise() * np.random.uniform(self.amplitude[0], self.amplitude[1])
            audio.data = clipData(data + matchDataToLen(noise, len(data)), 1)

            return audio

        def padNoise(self, noise, fs, time):
            if len(noise) < time * fs:
                noise = np.tile(noise, 1 + (time * fs // len(noise)))
            return noise

        def selNoise(self):
            return self.noises[np.random.randint(len(self.noises))]

    class whiteNoise(object):
        def __init__(self, amplitude=(-0.1, 0.1)):
            self.amplitude = amplitude

        def __call__(self, audio):
            audio.data = audio.data + np.random.uniform(self.amplitude[0], self.amplitude[1], size=audio.data.shape)
            return audio

    class clipAudio(object):
        def __init__(self, fraction):
            self.fraction = fraction
            
        def __call__(self, audio):
            nSamps = int(np.random.uniform(-self.fraction, self.fraction) * len(audio.data))

            if nSamps >= 0:
                audio.data = audio.data[abs(nSamps):]
            else:
                audio.data = audio.data[:-abs(nSamps)]

            return audio

    # Pad zeros randomly at left or right by a time or rate >= 0
    class padAudio(object):
        def __init__(self, fraction):
            self.fraction = fraction
            
        def __call__(self, audio):
            nSamps = int(np.random.uniform(-self.fraction, self.fraction) * len(audio.data))

            if nSamps >= 0:
                audio.data = np.concatenate((np.zeros(nSamps, ), audio.data))
            else:
                audio.data = np.concatenate((audio.data, np.zeros(-nSamps, )))

            return audio

    class amplifyAudio(object):
        def __init__(self, amplitude):
            self.amplitude = amplitude

        def __call__(self, audio):
            amp = np.random.uniform(self.amplitude[0], self.amplitude[1], size=audio.data.shape)
            audio.data = clipData(audio.data * amp, 1)
            return audio
