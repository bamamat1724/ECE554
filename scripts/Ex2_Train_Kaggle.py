
import sys, os
import torch

ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
sys.path.append(ROOT)

import common.dataset as dataset
import common.mods as mods
import common.lstm as lstm

args = lstm.set_default_args()

args.learning_rate = 0.001
args.num_epochs = 25
args.learning_rate_decay_interval = 5
args.learning_rate_decay_rate = 0.5
args.do_data_augment = True
args.train_eval_test_ratio=[0.9, 0.1, 0.0]
args.data_folder = "data/kaggle/"
args.classes_txt = "labels/classes_kaggle.names"
args.device = torch.device('cuda:0')
args.load_weights_from = None

files_name, files_label = dataset.load_filenames_and_labels(args.data_folder, args.classes_txt)

if args.do_data_augment:
    Mod = mods.Noiser
    modifiers = Mod([
        Mod.whiteNoise(amplitude=(-0.1, 0.1)),
        Mod.clipAudio(fraction=0.2),
        Mod.padAudio(fraction=0.2),
        Mod.amplifyAudio(amplitude=(0.2, 1.5)),
        Mod.recordedNoise(dir="data/noises/", prob_noise=0.7, amplitude=(0, 0.7))
    ], chance=0.8)
else:
    modifiers = None

tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y = lstm.split_train_eval_test(X=files_name, Y=files_label, ratios=args.train_eval_test_ratio, dtype='list')
train_dataset = dataset.AudioDataset(tr_X, tr_Y, transform=modifiers)
eval_dataset = dataset.AudioDataset(ev_X, ev_Y, transform=None)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, shuffle=True)
eval_loader = torch.utils.data.DataLoader(dataset=eval_dataset, shuffle=True)
model = lstm.create_RNN_model(args, load_weights_from=args.load_weights_from)
lstm.train_model(model, args, train_loader, eval_loader)
