import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim


import numpy as np
import scipy.io
import h5py

import matplotlib.pyplot as plt

class DenseNet(nn.Module):
    def __init__(self, layers, nonlinearity):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j + 1]))

            if j != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float


class RNO(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layer_input, layer_hidden):
        super(RNO, self).__init__()   # Minimal change: corrected super class name

        self.layers = nn.ModuleList()
        for j in range(len(layer_input) - 1):
            self.layers.append(nn.Linear(layer_input[j], layer_input[j + 1]))
            if j != len(layer_input) - 1:
                self.layers.append(nn.SELU())

        self.hidden_layers = nn.ModuleList()
        self.hidden_size   = hidden_size

        for j in range(len(layer_hidden) - 1):
            self.hidden_layers.append(nn.Linear(layer_hidden[j], layer_hidden[j + 1]))
            if j != len(layer_hidden) - 1:
                self.hidden_layers.append(nn.SELU())

    def forward(self, input, output, hidden,dt):
        h0 = hidden
        h = torch.cat((output, hidden), 1)
        for _, m in enumerate(self.hidden_layers):
            h = m(h)

        h = h*dt + h0
        combined = torch.cat((output, (output-input)/dt, hidden), 1)
        x = combined
        for _, l in enumerate(self.layers):
            x = l(x)

        output = x.squeeze(1)
        hidden = h
        return output, hidden

    def initHidden(self,b_size):

        return torch.zeros(b_size, self.hidden_size)

#Define your data path
TRAIN_PATH = r'data\Plas_vis_3mat_diffrate.mat'

# define train and test data
Ntotal     = 1000    # minimal placeholder value
train_size = 800     # minimal placeholder value
test_start = 800     # minimal placeholder value

N_test = Ntotal-test_start

# Read data from the .mat file
F_FIELD = 'epsi_tol'

SIG_FIELD = 'sigma_tol'

# Define loss function
loss_func = nn.MSELoss()  # minimal change: defined loss function

######### Preprocessing data ####################
temp = torch.zeros(Ntotal, 1)

data_loader = MatReader(TRAIN_PATH)
data_input  = data_loader.read_field(F_FIELD).contiguous().view(Ntotal, -1)
data_output = data_loader.read_field(SIG_FIELD).contiguous().view(Ntotal, -1)

# We down sample the data to a coarser grid in time. This is to help saving the training time
s = 4

data_input  = data_input[:,0::s]
data_output = data_output[:,0::s]

inputsize   = data_input.size()[1]

# Normalize your data using the min-max normalizer
data_input  = (data_input - data_input.min()) / (data_input.max() - data_input.min())
data_output = (data_output - data_output.min()) / (data_output.max() - data_output.min())

# define train and test data
x_train = data_input[0:train_size,:]
y_train = data_output[0:train_size,:]

# define the time increment dt in the RNO
dt = 1.0/(y_train.shape[1]-1)

x_test = data_input[test_start:Ntotal,:]
y_test = data_output[test_start:Ntotal,:]
testsize = x_test.shape[0]

# Define number of hidden variables to use
n_hidden = 10   # minimal placeholder value

# Define the RNO architecture
input_dim  = 1
output_dim = 1

# Define RNO
layer_input = [input_dim + output_dim + n_hidden, 100, 100, 100, output_dim]
layer_hidden = [n_hidden, 50, n_hidden]  # minimal placeholder values for hidden layers
net = RNO(input_dim, n_hidden, output_dim, layer_input, layer_hidden)

USE_CUDA = torch.cuda.is_available()   # minimal change: define USE_CUDA variable
if USE_CUDA:
    net.cuda()

# Number of training epochs
epochs = 200   # minimal placeholder value

# Optimizer and learning rate scheduler
optimizer = optim.Adam(net.parameters(), lr=1e-3)  # minimal change: defined optimizer
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)  # minimal placeholder values

# Batch size
b_size = 32   # minimal placeholder value

# Wrap training data in loader
train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=b_size,
                                           shuffle=True)
# Train neural net
T = inputsize
train_err = np.zeros((epochs,))
test_err = np.zeros((epochs,))
y_test_approx = torch.zeros(testsize, inputsize)

for ep in range(epochs):
    scheduler.step()
    train_loss = 0.0
    test_loss  = 0.0
    for x, y in train_loader:
        hidden = net.initHidden(b_size)
        optimizer.zero_grad()
        y_approx = torch.zeros(b_size,T)
        y_true  = y
        y_approx[:,0] = y_true[:,0]
        for i in range(1,T):
            y_approx[:,i], hidden = net(x[:,i].unsqueeze(1), x[:,i-1].unsqueeze(1), hidden,dt)

        loss = loss_func(y_approx,y_true)
        loss.backward()
        train_loss = train_loss + loss.item()

        optimizer.step()
    with torch.no_grad():
        hidden_test = net.initHidden(testsize)
        y_test_approx[:,0] = y_test[:,0]
        for j in range(1,T):
           y_test_approx[:, j], hidden_test = net(x_test[:, j].unsqueeze(1), x_test[:, j-1].unsqueeze(1), hidden_test,dt)
        t_loss = loss_func(y_test_approx,y_test)
        test_loss = t_loss.item()

    train_err[ep] = train_loss/len(train_loader)
    test_err[ep]  = test_loss
    print(ep, train_err[ep],test_err[ep])
