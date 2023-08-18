import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt



data = pd.read_csv('650_4000.csv', header=None)
training_set, test_set = train_test_split(data, test_size=0.3, random_state=0)
x_train = training_set.iloc[:,0:3474]
y_train = training_set.iloc[:,3474]
x_test = test_set.iloc[:,0:3474]
y_test = test_set.iloc[:,3474]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


x_train_tensor = torch.tensor(x_train.values, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
x_test_tensor = torch.tensor(x_test.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
print(y_train_tensor.shape)
# print(x_train_tensor.shape)

train_data = TensorDataset(x_train_tensor, y_train_tensor)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

test_data = TensorDataset(x_test_tensor, y_test_tensor)
test_loader = DataLoader(test_data, batch_size=32)

# Define the RCNN model

class RCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(RCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 64, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(128, 100, batch_first=True)
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(50, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x, _ = self.lstm(x.transpose(1, 2))
        x = x[:, -1, :]
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


num_classes = len(np.unique(y_train_tensor.numpy()))
input_size = x_train_tensor.shape[0]
model = RCNN(input_size, num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(100):
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        # target = torch.squeeze(target)
        target = target.to(device)

        # print(data.shape)
        # print(target.shape)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Evaluation
model.eval()
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data = data.to(device)
        # target = torch.squeeze(target)
        target = target.to(device)
        output = model(data)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

accuracy = correct / len(test_loader.dataset)
print(f"Test accuracy: {accuracy}")







# train_accuracies = []
# test_accuracies = []

# # Training loop
# for epoch in range(100):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data = data.to(device)
#         # target = torch.squeeze(target)
#         target = target.to(device)
#         optimizer.zero_grad()
#         output = model(data)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()

#     # Evaluate training accuracy
#     correct_train = 0
#     with torch.no_grad():
#         for data, target in train_loader:
#             data = data.to(device)
#         # target = torch.squeeze(target)
#             target = target.to(device)
#             output = model(data)
#             pred = output.argmax(dim=1, keepdim=True)
#             correct_train += pred.eq(target.view_as(pred)).sum().item()
#     train_accuracy = correct_train / len(train_loader.dataset)
#     train_accuracies.append(train_accuracy)

#     # Evaluate test accuracy
#     correct_test = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data = data.to(device)
#         # target = torch.squeeze(target)
#             target = target.to(device)
#             output = model(data)
#             pred = output.argmax(dim=1, keepdim=True)
#             correct_test += pred.eq(target.view_as(pred)).sum().item()
#     test_accuracy = correct_test / len(test_loader.dataset)
#     test_accuracies.append(test_accuracy)

# # Plot the training and test accuracies
# plt.plot(train_accuracies, label="Training Accuracy")
# plt.plot(test_accuracies, label="Test Accuracy")
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.show()