#  Week 1: Tensors and Operations on Tensors in PyTorch

# Autograd - automatic calculation of derivatives/partial derivatives via the chain rule for the backpropogation of error in neural network models. We use derivatives to change the model weights in a way that decreases loss. 
# Tensors hold on their gradients and update them. This will be useful in some model types where we want to accumulate gradients over multiple forward passes (e.g. recurrent neural networks which you will encounter soon) but for feedforward networks we want to reset the grads to zero after each forward pass. We can do this with: weights.grad=None
# If we want to update tensors for which requires_grad=True then we need to turn off the gradient computation by calling torch.no_grad() e.g.:with torch.no_grad(): w[0] = w[0] - learning_rate * w.grad[0]

import torch
import matplotlib.pyplot as plt

# Write code for a linear regression model predicting y from both features in X using Pytorch with autograd to obtain gradients. Print out the learning curve.

# Features. Two features and 13 data points; each row represents a feature, each column represents a data point, e.g. the first data point has features [-0.6832, -1.5407].
x=torch.tensor([[-0.6832,  0.2324, -1.2326, -0.3170,  0.3240, -1.2326, -1.5989,  0.7818,-0.3170,  0.2324,  1.0565,  1.4228,  1.3312],
[-1.5407, -1.2839, -1.0271, -0.7703, -0.5136, -0.2568,  0.0000,  0.2568, 0.5136,  0.7703,  1.0271,  1.2839,  1.5407]])

# Target values
y=torch.tensor([33,49,41,54,52,45,36,58,45,69,55,56,68])

# Visualise the features
plt.scatter(x[0], x[1], s=torch.exp(y/10), alpha=0.5)
print(plt.show())

# Fit a linear regression model
n_iters = 200
num_features = 2
weights = torch.rand(num_features,requires_grad=True)
bias=torch.tensor(0.0,requires_grad=True)
num_samples = len(y)
lr=0.005
linear_loss=[]

for i in range(n_iters):
    # Forward pass - calculate predicted y
    y_est = x[0] * weights[0] +x[1] * weights[1] + bias

    # Loss
    errors = y_est-y
    loss = errors.dot(errors)/num_samples
    linear_loss.append(loss.item()) # use items() to convert the loss tensor to a scalar for plotting the loss later

    # Backwards pass - calculate the gradient with respect to weights and bias
    loss.backward()

    # Update weights and bias with gradients (subtract gradient from weight)
    with torch.no_grad(): 
        weights -= lr * weights.grad
        bias -= lr * bias.grad

    # reset gradients after each pass
    weights.grad.zero_()
    bias.grad.zero_()

# Plotting the loss
plt.plot(linear_loss)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss over iterations')
plt.show()