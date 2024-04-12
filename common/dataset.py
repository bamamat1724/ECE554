import librosa, sys, os, torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.abspath(__file__))+"/../"
sys.path.append(ROOT)

import common.util as util


class AudioDataset(Dataset):
    def __init__(self, fNames, labels, transform=None):

        self.fNames, self.labels = fNames, labels
        self.labels = torch.tensor(self.labels, dtype=torch.int64)
        self.transform = transform

    def __len__(self):
        return len(self.fNames)

    def __getitem__(self, idx):
        filename = self.fNames[idx]
        audio = AudioClass(filename=filename)

        if self.transform:
            audio = self.transform(audio)

        audio.mfcc = librosa.feature.mfcc(n_mfcc=12, y=audio.data, sr=audio.fs, )
        inp = torch.tensor(audio.mfcc.T, dtype=torch.float32)
        output = self.labels[idx]

        return inp, output

def load_filenames_and_labels(data_folder, classes_txt):
    # Load classes
    with open(classes_txt, 'r') as f:
        classes = [l.rstrip() for l in f.readlines()]

    # Based on classes, load all filenames from data_folder
    files_name = []
    files_label = []
    for i, label in enumerate(classes):
        folder = data_folder + "/" + label + "/"

        names = util.files(folder, file_types="*.wav")
        labels = [i] * len(names)

        files_name.extend(names)
        files_label.extend(labels)

    print("Load data from: ", data_folder)
    print("\tClasses: ", ", ".join(classes))
    return files_name, files_label



class AudioClass(object):
    def __init__(self, filename=None, nCoeff=12):
        self.data, self.fs = util.read_audio(filename, dst_sample_rate=None)
        self.mfcc = None
        self.nCoeff = nCoeff
        self.filename = filename



