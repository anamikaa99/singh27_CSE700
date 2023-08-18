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



img_data_numpy = img_data.numpy()
img_data_torch = torch.tensor(img_data_numpy)
X_train = img_data_torch.permute(0, 3, 1, 2)
y_train = torch.tensor(target_val)

train_data = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)


img_data1_numpy = img_data1.numpy()
img_data1_torch = torch.tensor(img_data1_numpy)
X_test = img_data1_torch.permute(0, 3, 1, 2)
y_test = torch.tensor(target_val1)

test_data = TensorDataset(X_test, y_test)
test_loader = DataLoader(test_data, batch_size=32, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 3 * 200 * 200),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input).view(-1, 3, 200, 200)

# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Linear(3 * 200 * 200, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, input):
        input_flat = input.view(-1, 3 * 200 * 200)
        return self.main(input_flat)

# Hyperparameters
learning_rate = 0.0002
batch_size = 64
epochs = 500
latent_dim = 100

# Load your data here
# train_data = ...

# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

# Create the GAN components
generator = Generator().cuda()
discriminator = Discriminator().cuda()

# Loss and optimizers
criterion = nn.BCELoss()
optimizer_g = optim.Adam(generator.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)
G_losses = []
D_losses = []

# Training
for epoch in range(epochs):
    for i, data in enumerate(train_loader, 0):
        # Update discriminator
        discriminator.zero_grad()
        real_data = data[0].cuda()
        batch_size = real_data.size(0)
        real_label = torch.ones(batch_size, 1).cuda()
        fake_label = torch.zeros(batch_size, 1).cuda()

        output = discriminator(real_data)
        err_d_real = criterion(output, real_label)

        noise = torch.randn(batch_size, latent_dim).cuda()
        fake_data = generator(noise)
        output = discriminator(fake_data.detach())
        err_d_fake = criterion(output, fake_label)

        err_d = err_d_real + err_d_fake
        err_d.backward()
        optimizer_d.step()

        # Update generator
        generator.zero_grad()
        output = discriminator(fake_data)
        err_g = criterion(output, real_label)
        err_g.backward()
        optimizer_g.step()

        G_losses.append(err_g.item())
        D_losses.append(err_d.item())

    # Print and plot some generated images during training
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs} | D Loss: {err_d.item()} | G Loss: {err_g.item()}")
        with torch.no_grad():
            noise = torch.randn(64, latent_dim).cuda()
            fake = generator(noise).detach().cpu()
        plt.imshow(fake[0].permute(1, 2, 0))
        plt.show()

# Plot the D and G loss
plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label="G")
plt.plot(D_losses, label="D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()
