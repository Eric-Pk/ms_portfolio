import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pickle
# Load the dataset
with open('hw2_p3.pkl', 'rb') as file:
    data = pickle.load(file)
    # Assuming the data dictionary contains 'train' and 'test' as keys
    print(data[0])
    print(data[1])
    X_train = data[0]
    y_train = data[1]
    X_test = data[2]
    y_test = data[3]

class SimpleMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = SimpleMLP(input_dim=1000, output_dim=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


def train_model(model, criterion, optimizer, X_train, y_train, X_test, y_test, epochs=20):
    train_acc, test_acc = [], []
    for epoch in range(epochs):
        # Training phase
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        # Accuracy computation
        with torch.no_grad():
            model.eval()
            train_correct = (torch.argmax(model(X_train), dim=1) == y_train).sum().item()
            test_correct = (torch.argmax(model(X_test), dim=1) == y_test).sum().item()
            train_acc.append(train_correct / len(y_train))
            test_acc.append(test_correct / len(y_test))

    return train_acc, test_acc


# Assuming train and test datasets are prepared
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=64)

# Training the model
train_accuracy, test_accuracy = train_model(model, criterion, optimizer, X_train, y_train, X_test, y_test)

# Plotting
plt.plot(train_accuracy, label='Train Accuracy')
plt.plot(test_accuracy, label='Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
