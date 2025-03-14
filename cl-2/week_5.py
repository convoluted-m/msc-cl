import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil


x = np.linspace(0, 10, 2000)
y_true = np.sin(2*x) * 5 # sinewave
# Add some noise
y = y_true + np.random.normal(size=len(x)) * 2 # normally distrubuted noise, mean 0, sd = 2

plt.scatter(x, y_true, s = 0.5)
plt.scatter(x, y, s = 0.5)
plt.axvline(6, c = 'gray', linewidth = 0.5) # training dataset
plt.axvline(8, c = 'green', linewidth = 0.5); # dev set

# train the model, run on dev set, save model when it stops improving on dev set

idx = list(range(len(x)))

train_idx = [el for el in idx if x[el] < 6]
dev_idx = [el for el in idx if x[el] >= 6 and x[el] < 8]
test_idx = [el for el in idx if x[el] > 8]

# Example data
x_train = x[train_idx]; y_train = y[train_idx]
x_dev = x[dev_idx]; y_dev = y[dev_idx]
x_test = x[test_idx]; y_test = y[test_idx]

plt.scatter(x_train, y_train, s = 0.5)
plt.scatter(x_dev, y_dev, s = 0.5)
plt.scatter(x_test, y_test, s = 0.5)
plt.axvline(6, c = 'gray', linewidth = 0.5);
plt.axvline(8, c = 'green', linewidth = 0.5);

#torch.manual_seed(42) # to get the same numbers

# Define a basic multilayer perceptron (MLP)
# initialise the model randomly
class SimpleMLP(nn.Module):
    def __init__(self, hidden_dim=2048):
        super().__init__()
        #add layers; find balance before the model remembering stuff and it understanding stuff;
        # if too many layers we can overfit; start with a small hidden dimension and gradually increase it until results stop improving
        self.linear1= nn.Linear(1, hidden_dim)
        self.linear2= nn.Linear(hidden_dim, hidden_dim) # single hidden dimension
        self.linear3= nn.Linear(hidden_dim, 1)

    def forward(self, x): # take a single number, produce another single number
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = torch.relu(x)
        return self.linear3(x)
    

simple_mlp = SimpleMLP() #initialise the object

batch_size = 32
with torch.no_grad():
  result = simple_mlp(
      torch.tensor(x_train[:batch_size].reshape(-1, 1)).float() 
  )

# torch thinks 1st dimension is the batch dimension, 2nd is actual values
# need to make sure th input to the model is a matrix

plt.plot(result.numpy())

## Train
# Hyperparameterrs for training
# hidden dimension
# batch size
# learning rate - start with the biggest lr you think it would work, and if training doesn go well. you decrease it (if loss is not going down)
# choice of optimiser (but everybody uses AdamW)

simple_mlp = SimpleMLP()

batch_size = 32
learning_rate = 10**(-4)
optim = torch.optim.AdamW(list(simple_mlp.parameters()), lr=learning_rate)
# iterate over batches, how many batches we will have in the epoch
n_steps = ceil(len(x_train)/ batch_size) # len(x_train) + 1 //batch_size
n_epochs = 5 
loss_function = nn.MSELoss()

for epoch_n in tqdm(range(n_epochs)):
  train_losses = torch.zeros(n_steps)
  for step_n in tqdm(range(n_steps), leave=False,
                     desc=f'{epoch_n+1}, training'):
    optim.zero_grad()
    lo = step_n * batch_size
    hi = lo + batch_size
    input_x = torch.tensor(x_train[lo : hi].float().reshape(-1, 1)) 
    pred_output - simple_mlp(input_x)
    gold_output = torch.tensor(y_train[lo : hi].float().reshape(-1, 1))
    loss = loss_function(pred_output, gold_output)
    loss.backward()
    optim.step()

    train_losses[step_n] = loss.item()

  # The evaluation
  with torch.no_grad():
    dev_losses = torch.zeros(n_steps)

  print(f'Epoch{epoch_n+1} train loss: {train_losses.mean().item()}') # mean squared error

