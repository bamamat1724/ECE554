import numpy as np
import types
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import sys, os

ROOT = os.path.dirname(os.path.abspath(__file__))+"/../" # root of the project
sys.path.append(ROOT)

import common.util as util

def set_default_args():
    
    args = types.SimpleNamespace()

    # model params
    args.input_size = 12  # == n_mfcc
    args.batch_size = 1
    args.hidden_size = 32
    args.num_layers = 2

    # training params
    args.num_epochs = 100
    args.learning_rate = 0.0001
    args.learning_rate_decay_interval = 5 # decay for every 5 epochs
    args.learning_rate_decay_rate = 0.5 # lr = lr * rate
    args.weight_decay = 0.00
    args.gradient_accumulations = 16 # number of gradient accums before step
    
    # training params2
    args.load_weights_from = None
    args.finetune_model = False # If true, fix all parameters except the fc layer
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # data
    args.data_folder = "data/data_train/"
    args.train_eval_test_ratio=[0.9, 0.1, 0.0]
    args.do_data_augment = False

    # labels
    args.classes_txt = "config/classes.names" 
    args.num_classes = None # should be added with a value somewhere, like this:

    # log setting
    args.plot_accu = True # if true, plot accuracy for every epoch
    args.show_plotted_accu = False # if false, not calling plt.show(), so drawing figure in background
    args.save_model_to = 'checkpoints/' # Save model and log file
        #e.g: model_001.ckpt, log.txt, log.jpg
    
    return args 

def load_weights(model, weights, PRINT=False):
    # Load weights into model.
    # If param's name is different, raise error.
    # If param's size is different, skip this param.
    # see: https://discuss.pytorch.org/t/how-to-load-part-of-pre-trained-model/1113/2
    
    for i, (name, param) in enumerate(weights.items()):
        model_state = model.state_dict()
        
        if name not in model_state:
            print("-"*80)
            print("weights name:", name) 
            print("RNN states names:", model_state.keys()) 
            assert 0, "Wrong weights file"
            
        model_shape = model_state[name].shape
        if model_shape != param.shape:
            print(f"\nWarning: Size of {name} layer is different between model and weights. Not copy parameters.")
            print(f"\tModel shape = {model_shape}, weights' shape = {param.shape}.")
        else:
            model_state[name].copy_(param)
        
def create_RNN_model(args, load_weights_from=None):
    ''' A wrapper for creating a 'class RNN' instance '''
    
    
    # Update some dependent args
    args.num_classes = len(util.read_list(args.classes_txt)) # read from "config/classes.names"
    args.save_log_to = args.save_model_to + "log.txt"
    args.save_fig_to = args.save_model_to + "fig.jpg"
    
    # Create model
    device = args.device
    model = RNN(args.input_size, args.hidden_size, args.num_layers, args.num_classes, device).to(device)
    
    # Load weights
    if load_weights_from:
        print(f"Load weights from: {load_weights_from}")
        weights = torch.load(load_weights_from)
        load_weights(model, weights)
    
    return model

# Recurrent neural network (many-to-one)
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes, device, classes=None):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.device = device
        self.classes = classes

    def forward(self, x):
        # Set initial hidden and cell states
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) 
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(self.device) 
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # shape = (batch_size, seq_length, hidden_size)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

    def predict(self, x):
        '''Predict one label from one sample's features'''
        # x: feature from a sample, LxN
        #   L is length of sequency
        #   N is feature dimension
        x = torch.tensor(x[np.newaxis, :], dtype=torch.float32)
        x = x.to(self.device)
        outputs = self.forward(x)
        _, predicted = torch.max(outputs.data, 1)
        predicted_index = predicted.item()
        return predicted_index
    
    def set_classes(self, classes):
        self.classes = classes 
    
    def predict_audio_label(self, audio):
        idx = self.predict_audio_label_index(audio)
        assert self.classes, "Classes names are not set. Don't know what audio label is"
        label = self.classes[idx]
        return label

    def predict_audio_label_index(self, audio):
        audio.compute_mfcc()
        x = audio.mfcc.T # (time_len, feature_dimension)
        idx = self.predict(x)
        return idx
    
def evaluate_model(model, eval_loader, num_to_eval=-1):
    ''' Eval model on a dataset '''
    device = model.device
    correct = 0
    total = 0
    for i, (featuress, labels) in enumerate(eval_loader):

        featuress = featuress.to(device) # (batch, seq_len, input_size)
        labels = labels.to(device)

        # Predict
        outputs = model(featuress)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # stop
        if i+1 == num_to_eval:
            break
    eval_accu = correct / total
    print('  Evaluate on eval or test dataset with {} samples: Accuracy = {}%'.format(
        i+1, 100 * eval_accu)) 
    return eval_accu

def fix_weights_except_fc(model):
    not_fix = "fc"
    for name, param in model.state_dict().items():
        if not_fix in name:
            continue
        else:
            print(f"Fix {name} layer", end='. ')
            param.requires_grad = False
    print("")

def train_model(model, args, train_loader, eval_loader):

    device = model.device
    if args.finetune_model:
        fix_weights_except_fc(model)
        
    # -- create folder for saving model
    if args.save_model_to:
        if not os.path.exists(args.save_model_to):
            os.makedirs(args.save_model_to)
            
    # -- Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    optimizer.zero_grad()

    # -- For updating learning rate
    def update_lr(optimizer, lr):    
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    # -- Train the model
    total_step = len(train_loader)
    curr_lr = args.learning_rate
    cnt_batches = 0
    for epoch in range(1, 1+args.num_epochs):
        cnt_correct, cnt_total = 0, 0
        for i, (features, labels) in enumerate(train_loader):
            cnt_batches += 1

            ''' original code of pytorch-tutorial:
            images = images.reshape(-1, sequence_length, input_size).to(device)
            labels = labels.to(device)
            # we can see that the shape of images should be: 
            #    (batch_size, sequence_length, input_size)
            '''
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward() # error
            if cnt_batches % args.gradient_accumulations == 0:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # Record result
            _, argmax = torch.max(outputs, 1)
            cnt_correct += (labels == argmax.squeeze()).sum().item()
            cnt_total += labels.size(0)
            
            # Print accuracy
            train_accu = cnt_correct/cnt_total
            if (i+1) % 50 == 0 or (i+1) == len(train_loader):
                print ('Epoch [{}/{}], Step [{}/{}], Loss = {:.4f}, Train accuracy = {:.2f}' 
                    .format(epoch, args.num_epochs, i+1, total_step, loss.item(), 100*train_accu))
            continue
        print(f"Epoch {epoch} completes")
        
        # -- Decay learning rate
        if (epoch) % args.learning_rate_decay_interval == 0:
            curr_lr *= args.learning_rate_decay_rate # lr = lr * rate
            update_lr(optimizer, curr_lr)
    
        # -- Evaluate and save model
        if (epoch) % 1 == 0 or (epoch) == args.num_epochs:
            eval_accu = evaluate_model(model, eval_loader, num_to_eval=-1)
            if args.save_model_to:
                name_to_save = args.save_model_to + "/" + "{:03d}".format(epoch) + ".ckpt"
                torch.save(model.state_dict(), name_to_save)
                print("Save model to: ", name_to_save)

            if args.plot_accu and epoch == 1:
                plt.figure(figsize=(10, 8))
                plt.ion()
                if args.show_plotted_accu:
                    plt.show()
            if (epoch == args.num_epochs) or (args.plot_accu and epoch>1):
                if args.show_plotted_accu:
                    plt.pause(0.01) 
                plt.savefig(fname=args.save_fig_to)
        
        # An epoch end
        print("-"*80 + "\n")
    
    # Training end
    return
            
def split_train_eval_test(X, Y, ratios=[0.8, 0.1, 0.1], dtype='list'):
    X1, Y1, X2, Y2 = split_train_test(
        X, Y,
        1 - ratios[0],
        dtype=dtype, if_print=False)

    X2, Y2, X3, Y3 = split_train_test(
        X2, Y2,
        ratios[2] / (ratios[1] + ratios[2]),
        dtype=dtype, if_print=False)

    r1, r2, r3 = 100 * ratios[0], 100 * ratios[1], 100 * ratios[2]
    n1, n2, n3 = len(Y1), len(Y2), len(Y3)
    print(f"Split data into [Train={n1} ({r1}%), Eval={n2} ({r2}%),  Test={n3} ({r3}%)]")
    tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y = X1, Y1, X2, Y2, X3, Y3
    return tr_X, tr_Y, ev_X, ev_Y, te_X, te_Y


def split_train_test(X, Y, test_size=0, USE_ALL=False, dtype='numpy', if_print=True):
    assert dtype in ['numpy', 'list']

    def _print(s):
        if if_print:
            print(s)

    _print("split_train_test:")
    if dtype == 'numpy':
        _print("\tData size = {}, feature dimension = {}".format(X.shape[0], X.shape[1]))
        if USE_ALL:
            tr_X = np.copy(X)
            tr_Y = np.copy(Y)
            te_X = np.copy(X)
            te_Y = np.copy(Y)
        else:
            f = sklearn.model_selection.train_test_split
            tr_X, te_X, tr_Y, te_Y = f(X, Y, test_size=test_size, random_state=14123)
    elif dtype == 'list':
        _print("\tData size = {}, feature dimension = {}".format(len(X), len(X[0])))
        if USE_ALL:
            tr_X = X[:]
            tr_Y = Y[:]
            te_X = X[:]
            te_Y = Y[:]
        else:
            N = len(Y)
            train_size = int((1 - test_size) * N)
            randidx = np.random.permutation(N)
            n1, n2 = randidx[0:train_size], randidx[train_size:]

            def get(arr_vals, arr_idx):
                return [arr_vals[idx] for idx in arr_idx]

            tr_X = get(X, n1)[:]
            tr_Y = get(Y, n1)[:]
            te_X = get(X, n2)[:]
            te_Y = get(Y, n2)[:]
    _print("\tNum training: {}".format(len(tr_Y)))
    _print("\tNum evaluation: {}".format(len(te_Y)))
    return tr_X, tr_Y, te_X, te_Y