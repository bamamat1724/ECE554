#import vlc
import time
import random
import glob as glob
import librosa

import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__)) + "/../"  # root of the project
sys.path.append(ROOT)

import common.record_audio as rec
import common.lstm as lstm
import common.util as util


def play_media(media_path):
    instance = vlc.Instance()
    player = instance.media_player_new()
    media = instance.media_new(media_path)

    print(media_path)
    player.set_media(media)
    player.play()

    return player

def video_audio_files():
    files = []
    for file in glob.glob(r'./videos/*'):
        files.append(file)
    return files

def player_con(media_files,operation,index):
    if operation=="left":
        if index < 0 :
            index = len(media_files)-1
        else:
            index = index-1  

        media_path = media_files[index]  # Replace with your file path
        player = play_media(media_path)
    elif operation=="right":
        if index >= len(media_files)-1:
            index = 0
        else:
            index = index+1        
        media_path = media_files[index]  # Replace with your file path
        player = play_media(media_path)
    elif operation=="wow":
        index_1 = random.randint(0,len(media_files)-1)
        media_path = media_files[index_1]  # Replace with your file path
        player = play_media(media_path)
    else:
        index = 0
        media_path = media_files[index]  # Replace with your file path
        player = play_media(media_path)
    return player,index

def main():

    PATH_TO_WEIGHTS = "checkpoints_64x3/025.ckpt"
    PATH_TO_CLASSES = "labels/classes_kaggle.names"

    print('Creating LSTM...')
    model_args = lstm.set_default_args()
    model = lstm.create_RNN_model(model_args, PATH_TO_WEIGHTS)

    classes = util.read_list(PATH_TO_CLASSES)
    model.set_classes(classes)

    # Set up audio recorder
    recorder = rec.AudioRecorder()

    media_files = video_audio_files()
    print(len(media_files))
    operation = None
    index=0
    vc_enable = False
    #player,index = player_con(media_files,operation,index)

    while True:
        print('Listening...')
        # start recording
        recorder.start_record(folder='./data/data_tmp/')

        # Do anything here while recording
        # time.sleep(5)

        # stop recording
        recorder.stop_record()
        data, sample_rate = util.read_audio(recorder.filename, dst_sample_rate=None)
        mfcc = librosa.feature.mfcc(
            y=data,
            sr=sample_rate,
            n_mfcc=12,  # How many mfcc features to use? 12 at most.
            # https://dsp.stackexchange.com/questions/28898/mfcc-significance-of-number-of-features
        )
        operation = model.predict_audio_label(mfcc).lower()
        print(operation)

        if operation == "backward":
            if vc_enable:
                player.set_time(player.get_time() - 10000)  # Rewind 10 seconds
        elif operation == "down":
            if vc_enable:
                player.audio_set_volume(player.audio_get_volume() - 10)  # Volume down
        elif operation == "forward":
            if vc_enable:
                player.set_time(player.get_time() + 30000)  # Fast forward 30 seconds
        elif operation == "go":
            if vc_enable:
                player.play()  # Play
        elif operation == "left":
            if vc_enable:
                player.stop()
                player,index = player_con(media_files,operation,index)  # Previous episode
        elif operation == "marvin":
            print("Voice commands activated")  # Activate voice commands
            vc_enable = True
        elif operation == "right":
            if vc_enable:
                player.stop()
                player,index = player_con(media_files,operation,index) # Previous next episode
        elif operation == "stop":
            if vc_enable:
                player.pause()  # Pause
        elif operation == "up":
            if vc_enable:
                player.audio_set_volume(player.audio_get_volume() + 10)  # Volume up
        elif operation == "wow":
            if vc_enable:
                player.stop()
                player,index = player_con(media_files,operation,index)  # Random episode
        elif operation == "zero":
            if vc_enable:
                player.stop()  # Restart episode
                player.play()
        else:
            vc_enable = False

        time.sleep(0.3)  # Wait for 0.3 seconds before accepting the next operation

if __name__ == "__main__":
    main()
