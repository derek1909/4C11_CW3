import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim

import numpy as np
import scipy.io
import h5py
import os

import matplotlib.pyplot as plt

import wandb
from tqdm import tqdm

# ========================= CONFIGURATION ==========================
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# Model and training settings
wandb.init(project="RNO_project", config={
    "Use_ViscoElas_RNO": True,
    "Ntotal": 400,
    "train_size": 320,
    "downsampling": 4,
    "n_hidden": 0,
    "input_dim": 1,
    "output_dim": 1,
    "layer_input": [20, 20],
    "layer_hidden": [10],
    "epochs": 500,
    "learning_rate": 3e-2,
    "step_size": 50,
    "gamma": 0.7,
    "b_size": 80,
    "early_stop_patience": 100,
    "min_delta": 1e-5
})
config = wandb.config


TRAIN_PATH = 'Coursework3/viscodata_3mat.mat'
F_FIELD    = 'epsi_tol'
SIG_FIELD  = 'sigma_tol'

input_layer_width = config.input_dim*2 if config.Use_ViscoElas_RNO else config.input_dim
layer_input = [input_layer_width + config.n_hidden] + config.layer_input + [config.output_dim]
layer_hidden= [config.input_dim + config.n_hidden] + config.layer_hidden + [config.n_hidden]

if config.Use_ViscoElas_RNO:
    model_name = f"RNO_Visco_{config.n_hidden}"
else:
    model_name = f"RNO_Standard_{config.n_hidden}"
result_dir = f"/scratches/kolmogorov_2/jd976/working/4C11_CW3/Coursework3/results/{model_name}"
os.makedirs(result_dir, exist_ok=True)

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
    """
    A unified Recurrent Neural Operator that can operate in two modes:
      - Standard: uses only `input` and `hidden`
      - Viscoelastic: uses `input`, `prev_input`, and `hidden`
    The mode is controlled by the boolean flag `use_visco`.
    """
    def __init__(self, input_size, hidden_size, output_size, layer_input, layer_hidden, use_visco=False):
        super(RNO, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.use_visco = use_visco
        
        # Build feedforward layers.
        # Expecting the first layer input dimension to match:
        #   input_size + hidden_size (if not viscoelastic)
        # or
        #   input_size*2 + hidden_size (if viscoelastic)
        self.layers = nn.ModuleList()
        for j in range(len(layer_input) - 1):
            self.layers.append(nn.Linear(layer_input[j], layer_input[j + 1]))
            if j != len(layer_input) - 1:
                self.layers.append(nn.SELU())
        
        # Build hidden state update layers.
        self.hidden_layers = nn.ModuleList()
        for j in range(len(layer_hidden) - 1):
            self.hidden_layers.append(nn.Linear(layer_hidden[j], layer_hidden[j + 1]))
            if j != len(layer_hidden) - 1:
                self.hidden_layers.append(nn.SELU())
    
    def forward(self, input, hidden, dt, prev_input=None):
        # Update hidden state
        h0 = hidden
        h = torch.cat((input, hidden), 1)
        for m in self.hidden_layers:
            h = m(h)
        h = h * dt + h0
        
        # Construct the feedforward input based on mode
        if self.use_visco:
            if prev_input is None:
                raise ValueError("prev_input must be provided when use_visco is True")
            x = torch.cat((input, (input - prev_input) / dt, hidden), 1)
            expected_dim = self.input_size * 2 + self.hidden_size
        else:
            x = torch.cat((input, hidden), 1)
            expected_dim = self.input_size + self.hidden_size
        
        # Ensure the feature dimension is correct
        assert x.shape[1] == expected_dim, \
            f"Expected x.shape[1] to be {expected_dim}, but got {x.shape[1]}"
        
        # Pass through the feedforward layers
        for l in self.layers:
            x = l(x)
        output = x.squeeze(1)
        
        return output, h
    
    def initHidden(self, b_size):
        return torch.zeros(b_size, self.hidden_size).to(device)

# Define loss function
loss_func = nn.MSELoss()

# Read data from the .mat file
data_loader = MatReader(TRAIN_PATH)
data_input  = data_loader.read_field(F_FIELD).contiguous().view(config.Ntotal, -1)  # (400, 1001)
data_output = data_loader.read_field(SIG_FIELD).contiguous().view(config.Ntotal, -1)

# Downsample data
data_input  = data_input[:, 0::config.downsampling]  # (400, 251)
data_output = data_output[:, 0::config.downsampling]
inputsize   = data_input.size()[1]

# Normalize data using min-max normalization
data_input  = (data_input - data_input.min()) / (data_input.max() - data_input.min())
data_output = (data_output - data_output.min()) / (data_output.max() - data_output.min())

# Define train and test data
x_train = data_input[0:config.train_size, :]
y_train = data_output[0:config.train_size, :]
dt = 1.0 / (y_train.shape[1] - 1)
x_test = data_input[config.train_size:config.Ntotal, :]
y_test = data_output[config.train_size:config.Ntotal, :]
testsize = x_test.shape[0]

# Define RNO and move model to device
net = RNO(config.input_dim, config.n_hidden, config.output_dim, layer_input, layer_hidden, use_visco = config.Use_ViscoElas_RNO)
net = torch.compile(net)
net.to(device)

# Optimizer and learning rate scheduler
optimizer = optim.Adam(net.parameters(), lr=config.learning_rate)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.step_size, gamma=config.gamma)

# Wrap training data in loader
train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(x_train, y_train),
    batch_size=config.b_size,
    shuffle=True
)

# Prepare for training
T = inputsize
train_err = np.zeros((config.epochs,))
test_err = np.zeros((config.epochs,))
y_test_approx = torch.zeros(testsize, T, device=device)

# Move test data to device
x_test = x_test.to(device)
y_test = y_test.to(device)

# Early stopping parameters
best_loss = float('inf')
patience_counter = 0

# Train neural network
print(f"number of internal variables: {config.n_hidden}")
with tqdm(total=config.epochs, initial=0, desc="Training", unit="epoch", dynamic_ncols=True) as pbar_epoch:
    for ep in range(config.epochs):
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
                if net.use_visco:
                    y_approx[:, i], hidden = net(x[:, i].unsqueeze(1), hidden, dt, prev_input = x[:, i-1].unsqueeze(1))
                else:
                    y_approx[:, i], hidden = net(x[:, i].unsqueeze(1), hidden, dt)
            loss = loss_func(y_approx, y_true)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        with torch.no_grad():
            hidden_test = net.initHidden(testsize)
            y_test_approx[:, 0] = y_test[:, 0]
            for j in range(1, T):
                if net.use_visco:
                    y_test_approx[:, j], hidden_test = net(x_test[:, j].unsqueeze(1), hidden_test, dt, prev_input = x_test[:, j-1].unsqueeze(1))
                else:
                    y_test_approx[:, j], hidden_test = net(x_test[:, j].unsqueeze(1), hidden_test, dt)
            t_loss = loss_func(y_test_approx, y_test)
            test_loss = t_loss.item()
            # Compute RMSE and then the percentage error
            rmse = torch.sqrt(t_loss)
            test_error_percentage = rmse.item() * 100

        scheduler.step()
        avg_train_loss = train_loss / len(train_loader)
        train_err[ep] = avg_train_loss
        test_err[ep]  = test_loss
        wandb.log({
            "epoch": ep, 
            "train_loss": avg_train_loss, 
            "test_loss": test_loss,
            "test_error_percentage": test_error_percentage
        })
        pbar_epoch.update(1)
        pbar_epoch.set_postfix({
            "Train Loss": f"{avg_train_loss:.3e}", 
            "Test Loss": f"{test_loss:.3e}",
            "Test Err %": f"{test_error_percentage:.2f}%"
        })

        # Early stopping check
        if test_loss < best_loss - config.min_delta:
            best_loss = test_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.early_stop_patience:
            print(f"Early stopping triggered at epoch {ep}.")
            wandb.log({"early_stop_epoch": ep})
            break

torch.save(net.state_dict(), os.path.join(result_dir, f"{model_name}.pt"))
np.savez(os.path.join(result_dir, "training_history.npz"), train_err=train_err, test_err=test_err)

# Plot training history and save to local machine (do not show the plot)
plt.figure()
plt.plot(np.arange(config.epochs), train_err, label="Train Loss")
plt.plot(np.arange(config.epochs), test_err, label="Test Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.savefig(os.path.join(result_dir, "training_history.png"))
plt.close()

wandb.log({"training_history": wandb.Image("training_history.png")})