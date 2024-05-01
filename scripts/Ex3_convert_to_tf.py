import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import onnx
from onnx_tf.backend import prepare


trained_model = Net()
trained_model.load_state_dict(torch.load('./checkpoints_64x3/025.ckpt'))
dummy_input = Variable(torch.randn(1, 1, 28, 28))
torch.onnx.export(trained_model, dummy_input, "lstm.onnx")

model = onnx.load('lstm.onnx')
tf_rep = prepare(model)

tf_rep.export_graph('lstm.pb')