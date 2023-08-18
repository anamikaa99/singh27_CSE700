# !pip install -qq -U tensorflow-addons
import math
import ntpath
import cv2
import os
import numpy as np
import pandas as pd
import splitfolders 
from sklearn.metrics import confusion_matrix
import seaborn as sns
from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, TensorDataset

import time
import copy

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


# class_labels = {name: idx for idx, name in enumerate(set(class_name))}
# class_name = [class_labels[label] for label in class_name]

img_data_numpy = img_data.numpy()
img_data_torch = torch.tensor(img_data_numpy)
X_train = img_data_torch.permute(0, 3, 1, 2)
y_train = torch.tensor(target_val)

# Create a DataLoader
train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


img_data1_numpy = img_data1.numpy()
img_data1_torch = torch.tensor(img_data1_numpy)
X_test = img_data1_torch.permute(0, 3, 1, 2)
y_test = torch.tensor(target_val1)

# Create a DataLoader
test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)

NUM_CLASSES = 5





# Load ResNet model and modify the last layer for 5 classes

model = models.resnet18(pretrained=True)#.to(device)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_loss_list = []
train_accuracy_list = []


# Train the model
for epoch in range(10): 
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        # images = images.to(device)
        # labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()

    
    avg_train_loss = train_loss / len(train_loader)
    train_loss_list.append(avg_train_loss)

    
    train_accuracy = 100 * correct_train / total_train
    train_accuracy_list.append(train_accuracy)

    print(f"Epoch {epoch + 1} loss: {avg_train_loss}, training accuracy: {train_accuracy}%")

# Evaluate the model 
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        # images = images.to(device)
        # labels = labels.to(device)
        outputs = model(images.float())
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Accuracy: {100 * correct / total}%")



plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_loss_list)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1, 2, 2)
plt.plot(train_accuracy_list)
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')

plt.tight_layout()
plt.show()

