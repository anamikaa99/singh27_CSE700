import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.metrics import accuracy_score


data = pd.read_csv('650_4000.csv', header=None)

training_set, test_set = train_test_split(data, test_size=0.3, random_state=0)
x_train = training_set.iloc[:,0:3474]
y_train = training_set.iloc[:,3474]
x_test = test_set.iloc[:,0:3474]
y_test = test_set.iloc[:,3474]

# Apply PCA
n_components = 500 
pca = PCA(n_components=n_components)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

# Train Gaussian Process Classifier on the reduced dataset
kernel = 1.0 * RBF(1.0)
gpc = GaussianProcessClassifier(kernel=kernel, random_state=0).fit(x_train_pca, y_train)

train_accuracy = gpc.score(x_train_pca, y_train)
test_accuracy = gpc.score(x_test_pca, y_test)

print(f"Train accuracy: {train_accuracy}")
print(f"Test accuracy: {test_accuracy}")

