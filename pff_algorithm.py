# pff_algorithm.py

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# Define the neural network
class FFNN(nn.Module):
    def __init__(self, input_size, hidden_sizes):
        super(FFNN, self).__init__()
        layers = []
        last_size = input_size
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Goodness function
def goodness(x):
    return torch.sum(x ** 2, dim=1)

# Forward-Forward algorithm
def forward_forward(model, x_pos, x_neg, threshold):
    pos_output = model(x_pos)
    neg_output = model(x_neg)
    pos_goodness = goodness(pos_output)
    neg_goodness = goodness(neg_output)
    
    pos_loss = torch.mean((pos_goodness - threshold) ** 2)
    neg_loss = torch.mean((neg_goodness - threshold) ** 2)
    
    return pos_loss + neg_loss

# Training function for FF algorithm
def train_ff(model, data_loader, epochs, threshold, optimizer):
    model.train()
    for epoch in range(epochs):
        for x, _ in data_loader:
            x = x.view(x.size(0), -1)
            
            # Generate negative samples by shuffling
            x_neg = x[torch.randperm(x.size(0))]
            
            optimizer.zero_grad()
            loss = forward_forward(model, x, x_neg, threshold)
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# Dataset and DataLoader
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Split dataset into training and validation
train_size = int(0.8 * len(mnist_dataset))
val_size = len(mnist_dataset) - train_size
train_dataset, val_dataset = random_split(mnist_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Model, optimizer, and threshold
input_size = 784
hidden_sizes = [2000, 2000, 2000, 2000]
model = FFNN(input_size, hidden_sizes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
threshold = 0.1

# Train the model
train_ff(model, train_loader, epochs=10, threshold=threshold, optimizer=optimizer)

# Single-Layer PFF training function
def train_single_layer_pff(layers, data_loader, epochs, threshold, optimizers):
    for i, layer in enumerate(layers):
        print(f"Training Layer {i+1}")
        for epoch in range(epochs):
            for x, _ in data_loader:
                x = x.view(x.size(0), -1)
                x_neg = x[torch.randperm(x.size(0))]
                
                optimizer = optimizers[i]
                optimizer.zero_grad()
                
                # Forward pass through the current layer
                pos_output = layer(x)
                neg_output = layer(x_neg)
                
                pos_goodness = goodness(pos_output)
                neg_goodness = goodness(neg_output)
                
                pos_loss = torch.mean((pos_goodness - threshold) ** 2)
                neg_loss = torch.mean((neg_goodness - threshold) ** 2)
                
                loss = pos_loss + neg_loss
                loss.backward()
                optimizer.step()
            
            print(f"Epoch {epoch + 1}/{epochs}, Layer {i+1} Loss: {loss.item()}")

# Splitting the model into layers
layers = [nn.Sequential(*model.net[:i]) for i in range(2, len(model.net) + 1, 2)]
optimizers = [optim.Adam(layer.parameters(), lr=0.001) for layer in layers]

# Train the model with Single-Layer PFF
train_single_layer_pff(layers, train_loader, epochs=10, threshold=threshold, optimizers=optimizers)
