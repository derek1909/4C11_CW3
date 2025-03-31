import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import numpy as np
import scipy.io
import h5py

import matplotlib.pyplot as plt

import wandb
from tqdm import tqdm

# ========================= CONFIGURATION ==========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data settings
TRAIN_PATH = 'Coursework3/viscodata_3mat.mat'
Ntotal     = 400   
train_size = 300
test_size  = Ntotal - train_size

F_FIELD    = 'epsi_tol'
SIG_FIELD  = 'sigma_tol'
s          = 4  # downsampling factor

# Model and training settings
n_hidden    = 0
input_dim   = 1
output_dim  = 1
layer_input = [input_dim*2 + n_hidden, 100, 100, 100, output_dim]
layer_hidden= [input_dim + n_hidden, 50, n_hidden]

epochs       = 200
learning_rate= 1e-3
step_size    = 50
gamma        = 0.8
b_size       = 100

# Initialize wandb
wandb.init(project="RNO_project", config={
    "Ntotal": Ntotal,
    "train_size": train_size,
    "test_size": test_size,
    "s": s,
    "n_hidden": n_hidden,
    "input_dim": input_dim,
    "output_dim": output_dim,
    "layer_input": layer_input,
    "layer_hidden": layer_hidden,
    "epochs": epochs,
    "learning_rate": learning_rate,
    "step_size": step_size,
    "gamma": gamma,
    "b_size": b_size
})

# ==================================================================


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
            x = np.transpose(x, axes=range(len(x.shape)-1, -1, -1))
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
    # RNO: Recurrent Neural Operator
    def __init__(self, input_size, hidden_size, output_size, layer_input, layer_hidden):
        super(RNO, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        # Initialize the feedforward layers based on the architecture defined in layer_input.
        self.layers = nn.ModuleList()
        for j in range(len(layer_input) - 1):
            self.layers.append(nn.Linear(layer_input[j], layer_input[j + 1]))
            if j != len(layer_input) - 1:
                self.layers.append(nn.SELU())
        
        # Initialize the hidden state update layers based on the architecture in layer_hidden.
        self.hidden_layers = nn.ModuleList()
        for j in range(len(layer_hidden) - 1):
            self.hidden_layers.append(nn.Linear(layer_hidden[j], layer_hidden[j + 1]))
            if j != len(layer_hidden) - 1:
                self.hidden_layers.append(nn.SELU())
    
    def forward(self, input, prev_input, hidden, dt):
        h0 = hidden
        
        h = torch.cat((input, hidden), 1)
        for _, m in enumerate(self.hidden_layers):
            h = m(h)
        h = h * dt + h0

        x = torch.cat((input, (input - prev_input) / dt, hidden), 1)
        assert x.shape[1] == (self.input_size*2 + self.hidden_size), f"Expected x.shape[1] to be { (self.input_size*2 + self.hidden_size)}, but got {x.shape[1]}"
        for _, l in enumerate(self.layers):
            x = l(x)
        output = x.squeeze(1)

        hidden = h
        return output, hidden
    
    def initHidden(self, b_size):
        return torch.zeros(b_size, self.hidden_size).to(device)

# Define loss function
loss_func = nn.MSELoss()

# Read data from the .mat file
data_loader = MatReader(TRAIN_PATH)
data_input  = data_loader.read_field(F_FIELD).contiguous().view(Ntotal, -1)  # (400, 1001)
data_output = data_loader.read_field(SIG_FIELD).contiguous().view(Ntotal, -1)

# Downsample data
data_input  = data_input[:, 0::s]  # (400, 251)
data_output = data_output[:, 0::s]
inputsize   = data_input.size()[1]

# Normalize data using min-max normalization
data_input  = (data_input - data_input.min()) / (data_input.max() - data_input.min())
data_output = (data_output - data_output.min()) / (data_output.max() - data_output.min())

# Define train and test data
x_train = data_input[0:train_size, :]
y_train = data_output[0:train_size, :]
dt = 1.0 / (y_train.shape[1] - 1)
x_test = data_input[train_size:Ntotal, :]
y_test = data_output[train_size:Ntotal, :]
testsize = x_test.shape[0]

# Define RNO and move model to device
net = RNO(input_dim, n_hidden, output_dim, layer_input, layer_hidden)
net.to(device)

# Optimizer and learning rate scheduler
optimizer = optim.Adam(net.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# Wrap training data in loader
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=b_size,
    shuffle=True
)

# Prepare for training
T = inputsize
train_err = np.zeros((epochs,))
test_err = np.zeros((epochs,))
y_test_approx = torch.zeros(testsize, T, device=device)

# Move test data to device
x_test = x_test.to(device)
y_test = y_test.to(device)

# Train neural network
# Train neural network
with tqdm(total=epochs, initial=0, desc="Training", unit="epoch", dynamic_ncols=True) as pbar_epoch:
    for ep in range(epochs):
        train_loss = 0.0
        test_loss  = 0.0

        for x, y in train_loader:
            current_batch_size = x.size(0)
            x, y = x.to(device), y.to(device)
            hidden = net.initHidden(current_batch_size)
            optimizer.zero_grad()
            y_approx = torch.zeros(current_batch_size, T, device=device)
            y_true = y
            y_approx[:, 0] = y_true[:, 0]
            for i in range(1, T):
                y_approx[:, i], hidden = net(x[:, i].unsqueeze(1), x[:, i-1].unsqueeze(1), hidden, dt)
            loss = loss_func(y_approx, y_true)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        with torch.no_grad():
            hidden_test = net.initHidden(testsize)
            y_test_approx[:, 0] = y_test[:, 0]
            for j in range(1, T):
                y_test_approx[:, j], hidden_test = net(x_test[:, j].unsqueeze(1), x_test[:, j-1].unsqueeze(1), hidden_test, dt)
            t_loss = loss_func(y_test_approx, y_test)
            test_loss = t_loss.item()

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        train_err[ep] = avg_train_loss
        test_err[ep]  = test_loss
        wandb.log({"epoch": ep, "train_loss": avg_train_loss, "test_loss": test_loss})
        pbar_epoch.update(1)
        pbar_epoch.set_postfix({"Train Loss": f"{avg_train_loss:.3e}", "Test Loss": f"{test_loss:.3e}"})

# Plot training history and save to local machine (do not show the plot)
plt.figure()
plt.plot(np.arange(epochs), train_err, label="Train Loss")
plt.plot(np.arange(epochs), test_err, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.savefig("training_history.png")
plt.close()