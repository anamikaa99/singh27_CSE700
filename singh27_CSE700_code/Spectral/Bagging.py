import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('650_4000.csv', header=None)

training_set, test_set = train_test_split(data, test_size=0.3, random_state=0)
x_train = training_set.iloc[:,0:3474]
y_train = training_set.iloc[:,3474]
x_test = test_set.iloc[:,0:3474]
y_test = test_set.iloc[:,3474]

base_estimator = DecisionTreeClassifier(max_depth=2)

# Create and fit a Bagging classifier
bagging_clf = BaggingClassifier(base_estimator=base_estimator, n_estimators=200, random_state=42)
bagging_clf.fit(x_train, y_train)

# Predicting on test data
y_pred = bagging_clf.predict(x_test)

# Calculating accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc}")
