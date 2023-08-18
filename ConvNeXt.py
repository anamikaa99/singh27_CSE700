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
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model
from keras.layers import BatchNormalization
from tensorflow.keras.layers import LeakyReLU
from  matplotlib import pyplot as plt
import matplotlib.image as mpimg
import torch
import torch.optim as optim
import torch.nn.functional as F
from timm.models.layers import DropPath
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
y = target_val 
X1 = img_data1  # Image feature matrix (n_samples, n_features)
y1 = test_val  # Target variable (n_samples,)


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# Convert the selected features to a PyTorch tensor
tensor_features = torch.tensor(X, dtype=torch.float32)

# Convert the labels to a PyTorch tensor
tensor_labels = torch.tensor(y, dtype=torch.long)
test_features = torch.tensor(X1, dtype=torch.float32)
test_labels = torch.tensor(y1, dtype=torch.long)
tensor_features = tensor_features.permute(0,3,1,2)
test_features = test_features.permute(0,3,1,2)

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
testdataset = CustomDataset(test_features, test_labels)


batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(testdataset, batch_size=batch_size, shuffle=False)
device = ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

# !pip install -Uqq fastai timm


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, channel_multiplier=1, drop_path_rate=0.0):
        super(ConvNeXtBlock, self).__init__()

        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * channel_multiplier, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels * channel_multiplier),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * channel_multiplier, out_channels, kernel_size=3, stride=1, padding=1, groups=out_channels, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        x = self.shortcut(x) + self.drop_path(self.dwconv(x))
        return x




class ConvNeXtStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, channel_multiplier=1, drop_path_rate=0.0):
        super(ConvNeXtStage, self).__init__()

        if stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * channel_multiplier, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * channel_multiplier),
            )
        else:
            self.shortcut = nn.Identity()

        self.dwconv = nn.Sequential(
            nn.Conv2d(in_channels * channel_multiplier, in_channels * channel_multiplier, kernel_size=3, stride=stride, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(in_channels * channel_multiplier),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * channel_multiplier, out_channels * channel_multiplier, kernel_size=3, stride=1, padding=1, groups=16, bias=False),
            nn.BatchNorm2d(out_channels * channel_multiplier),
        )

        self.drop_path = DropPath(drop_path_rate)

    def forward(self, x):
        x = self.shortcut(x) + self.drop_path(self.dwconv(x))
        return x




class ConvNeXt(nn.Module):
    def __init__(self, num_classes=7):
        super(ConvNeXt, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.stages = nn.ModuleList([
        ConvNeXtStage(128, 256, 2),
        ConvNeXtStage(256, 512, 2),
        ConvNeXtStage(512, 1024, 2),
    ])


        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(1024, num_classes),
        )

        self.drop_path = DropPath(0.2)

    def forward(self, x):
        x = self.stem(x)
        for stage in self.stages:
            x = stage(x)
        x = self.drop_path(x)
        x = self.head(x)
        return x

def train(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_correct = 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).to(device)
            loss = criterion(outputs, labels).to(device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (outputs.argmax(1) == labels).sum().item()
        train_loss /= len(train_loader)
        train_accuracy = 100. * train_correct / len(train_loader.dataset)
        val_loss = 0.0
        val_correct = 0
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images).to(device)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            val_correct += (outputs.argmax(1) == labels).sum().item()
        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct / len(val_loader.dataset)
        print("Epoch: {} Train Loss: {:.4f} Train Acc: {:.4f} Val Loss: {:.4f} Val Acc: {:.4f}".format(
            epoch, train_loss, train_accuracy, val_loss, val_accuracy
        ))

if __name__ == "__main__":
    model = ConvNeXt().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(testdataset, batch_size=64, shuffle=False)
    num_epochs = 100
    train(model, criterion, optimizer, train_loader, val_loader, num_epochs)

