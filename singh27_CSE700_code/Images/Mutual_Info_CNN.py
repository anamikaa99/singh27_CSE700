# !pip install -qq -U tensorflow-addons
import math
import ntpath
import cv2
import os
import torch
import numpy as np
import pandas as pd
import splitfolders 
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import SelectKBest, mutual_info_classif

import seaborn as sns

import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from tensorflow.keras import layers

from keras import layers, Input
from keras.layers import InputLayer
from keras.layers.core import Dense, Flatten
from tensorflow.keras.models import Sequential, Model
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
%matplotlib inline
# from tensorflow.keras import layers, Dense, Input, InputLayer, Flatten

# Setting seed for reproducibiltiy
SEED = 42
keras.utils.set_random_seed(SEED)
##wheb using google colab

# from google.colab import drive
# drive.mount('/content/gdrive')
# !cp '/content/gdrive/MyDrive/CSE700/Images of Consumer Products.zip' .
# !unzip '/content/Images of Consumer Products.zip'

## splitting image folder into train and test
# !pip install split-folders

input_folder = "//content/Images of Consumer Products"
output = "/content/output" #where you want the split datasets saved. one will be created if it does not exist or none is set
splitfolders.ratio(input_folder, output=output, seed=42, ratio=(.9, .1)) # splitfolders.ratio('Data', output="output", seed=1337, ratio=(.8, 0.1,0.1))

IMG_WIDTH=200
IMG_HEIGHT=200
img_folder=r'/content/output/train'
val_folder = r'/content/output/val'
# test_folder=r'/content/output/test'


def create_dataset(folder):

    img_data_array=[]
    i = 0
    class_name=[]
    num_samples = 7546
    height = 200
    width = 200
    error_indices = []


    for dir1 in os.listdir(folder):
        for file in os.listdir(os.path.join(folder, dir1)):

          image_path= os.path.join(folder, dir1,  file)
          image= cv2.imread( image_path, cv2.COLOR_BGR2RGB)



          if ((image is not None) and len(image.shape) == 3):
            image= cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH),interpolation = cv2.INTER_AREA)
            image=np.array(image)
            image = image.astype('float32')
            image /= 255
            img_data_array.append(image)
            class_name.append(dir1)
            i = i+1


          else:
            print("Failed to load the image.")
            print(image_path)
            print(i)
            continue



    return img_data_array, class_name

# extract the image array and class name
img_data, class_name =create_dataset(img_folder)
img_data1, class_name1 =create_dataset(val_folder)
# img_data2, class_name2 =create_dataset(test_folder)

img_data = np.array(img_data)
img_data1 = np.array(img_data1)
# img_data2 = np.array(img_data2)

img_data = tf.convert_to_tensor(img_data)
img_data1 = tf.convert_to_tensor(img_data1)
# img_data2 = tf.convert_to_tensor(img_data2)


print(img_data.shape)
# print(type(img_data1.shape)


target_dict={k: v for v, k in enumerate(np.unique(class_name))}
target_val=  [target_dict[class_name[i]] for i in range(len(class_name))]
target_val1=  [target_dict[class_name1[i]] for i in range(len(class_name1))]
# target_val2=  [target_dict[class_name2[i]] for i in range(len(class_name2))]

X = img_data  # Image feature matrix (n_samples, n_features)
y = target_val  # Target variable (n_samples,)

# Convert images to 1D feature vectors if necessary
X = X.reshape(X.shape[0], -1)

# Perform mutual information feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=2500)  # Choose the number of top features (k)
X_new = selector.fit_transform(X, y)

# Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# Print the indices of the selected features
print("Selected feature indices:", selected_feature_indices)



tensor_features = torch.tensor(X_new, dtype=torch.float32)

# Convert the labels to a PyTorch tensor
tensor_labels = torch.tensor(y, dtype=torch.long)
print(tensor_features.shape)

# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# Create an instance of the custom dataset
dataset = CustomDataset(tensor_features, tensor_labels)





# Define the CNN architecture
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=1)
        self.fc = nn.Linear(79936, 5)  # Adjust the output size based on your classification task

    def forward(self, x):
        x = x.unsqueeze(1)  # Add a channel dimension to match the expected input shape of the CNN
        # torch.unsqueeze(x, 0)
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.relu(x)
        # print(x.shape)
        x = self.pool(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        # print(x.shape)
        x = self.fc(x)
        return x

# Create an instance of the CNN



# Create an instance of the CNN
cnn = CNN()

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(cnn.parameters(), lr=0.001)

# Create a data loader for batching
batch_size = 50
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    total_correct = 0
    total_samples = 0

    for batch_features, batch_labels in dataloader:
        # Forward pass
        # batch_features = batch_features.unsqueeze(1)
        outputs = cnn(batch_features)
        _, predicted_labels = torch.max(outputs, 1)

        total_samples += batch_labels.size(0)
        total_correct += (predicted_labels == batch_labels).sum().item()

        loss = criterion(outputs, batch_labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    accuracy = total_correct / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Training Accuracy: {accuracy:.4f}")


X1 = img_data1  # Image feature matrix (n_samples, n_features)
y1 = test_val  # Target variable (n_samples,)

# Convert images to 1D feature vectors if necessary
X1 = X1.reshape(X1.shape[0], -1)
# X1 = X1.numpy()

# Perform mutual information feature selection
selector = SelectKBest(score_func=mutual_info_classif, k=2500)  # Choose the number of top features (k)
X_new1 = selector.fit_transform(X1, y1)

# # Get the indices of the selected features
selected_feature_indices = selector.get_support(indices=True)

# # Print the indices of the selected features
print("Selected feature indices:", selected_feature_indices)


test_features = torch.tensor(X_new1, dtype=torch.float32)

# Convert the labels to a PyTorch tensor
test_labels = torch.tensor(y1, dtype=torch.long)
datasettest = CustomDataset(test_features, test_labels)
testdataloader = DataLoader(datasettest, batch_size=batch_size)


cnn.eval()
total_correct = 0
total_samples = 0

with torch.no_grad():
  for batch_features, batch_labels in testdataloader:
      outputs = cnn(batch_features)
      _, predicted_labels = torch.max(outputs, 1)

      total_samples += batch_labels.size(0)
      total_correct += (predicted_labels == batch_labels).sum().item()

accuracy = total_correct / total_samples
print(f"Test Accuracy: {accuracy:.4f}")