import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

data = pd.read_csv('650_4000.csv', header=None)

training_set, test_set = train_test_split(data, test_size=0.3, random_state=0)
x_train = training_set.iloc[:,0:3474]
y_train = training_set.iloc[:,3474]
x_test = test_set.iloc[:,0:3474]
y_test = test_set.iloc[:,3474]


bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

bdt.fit(x_train, y_train)

# Predicting on test data
y_pred = bdt.predict(x_test)

# Calculating accuracy
acc = accuracy_score(y_test, y_pred)
print(f"Test accuracy: {acc}")