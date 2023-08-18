import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


# Load the dataset
data = pd.read_csv('650_4000.csv', header=None)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Splitting data into training and test set
x = data.iloc[:, 0:3474].values
y = data.iloc[:, 3474].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

# Scaling data
t = MinMaxScaler()
t.fit(x_train)
x_train = t.transform(x_train)
x_test = t.transform(x_test)

# Converting to PyTorch tensors
from torch.utils.data import TensorDataset, DataLoader
import torch

train_data = TensorDataset(torch.Tensor(x_train), torch.Tensor(y_train))
test_data = TensorDataset(torch.Tensor(x_test), torch.Tensor(y_test))
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
test_loader = DataLoader(test_data, batch_size=128, shuffle=False)

# Add noise to the training data
x_train_noisy = x_train + 0.1 * np.random.randn(len(x_train), 3474)

# Define the denoising autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()

        # Define the encoder layers
        self.e1 = nn.Linear(3474, 1000)
        self.e2 = nn.Linear(1000, 500)
        self.bottleneck = nn.Linear(500, 250)

        # Define the decoder layers
        self.d1 = nn.Linear(250, 500)
        self.d2 = nn.Linear(500, 1000)
        self.output = nn.Linear(1000, 3474)

    def forward(self, x):
        x = F.relu(self.e1(x))
        x = F.relu(self.e2(x))
        x = F.relu(self.bottleneck(x))
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        return self.output(x)

# Create an instance of the denoising autoencoder model
model = DenoisingAutoencoder()

# Compile the denoising autoencoder model
optimizer = optim.Adam(model.parameters(), lr=0.001)


# Train the denoising autoencoder model
# model.train(x_train_noisy, x_train, epochs=10)
train_losses = []
model.train()
for epoch in range(20):
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        optimizer.zero_grad()
        recon_batch = model(data)
        loss = F.mse_loss(recon_batch, data)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_losses.append(train_loss)

    print(f'Epoch {epoch}: Loss {train_loss}')


plt.plot(train_losses)
plt.title('Train Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()


# Testing Loop
model.eval()
test_loss = 0
with torch.no_grad():
    for data, _ in test_loader:
        recon_batch = model(data)
        test_loss += F.mse_loss(recon_batch, data).item()

# test_loss /= len(test_loader.dataset)
print(f'Test loss: {test_loss}')


def extract_features(loader):
    features = []
    labels = []
    with torch.no_grad():
        for data, target in loader:
            h = model.e1(data)
            h = model.e2(h)
            h = model.bottleneck(h)
            features.append(h)
            labels.append(target)
    return torch.cat(features), torch.cat(labels)

train_features, train_labels = extract_features(train_loader)
test_features, test_labels = extract_features(test_loader)

train_features.shape


pca = PCA(n_components=20)
train_features_2d = pca.fit_transform(train_features)
test_features_2d = pca.transform(test_features)

# Fit k-means to the training data
kmeans = KMeans(n_clusters=5, random_state=42)
kmeans.fit(train_features_2d)

# Predict on the test data
test_predictions = kmeans.predict(test_features_2d)
silhouette_avg = silhouette_score(test_features_2d, test_predictions)
print(f'Silhouette Score: {silhouette_avg:.2f}')

# Plot the clusters
plt.scatter(test_features_2d[:, 0], test_features_2d[:, 1], c=test_predictions, cmap='viridis')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', marker='x')
plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar().set_label('Cluster ID')
plt.show()
