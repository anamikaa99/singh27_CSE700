import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import lightgbm as lgb


data = pd.read_csv('650_4000.csv', header=None)

training_set, test_set = train_test_split(data, test_size=0.3, random_state=0)

x_train = training_set.iloc[:,0:3474]
y_train = training_set.iloc[:,3474]
x_test = test_set.iloc[:,0:3474]
y_test = test_set.iloc[:,3474]

# !pip install lightgbm


train_data = lgb.Dataset(x_train, label=y_train)
test_data = lgb.Dataset(x_test, label=y_test, reference=train_data)

params = {
    'objective': 'multiclass',
    'num_class': 5, 
}

# Train the model
num_round = 100
bst = lgb.train(params, train_data, num_round, valid_sets=[test_data])

# Predict
test_pred = np.argmax(bst.predict(x_test), axis=1)

test_accuracy = accuracy_score(y_test, test_pred)

print(f"Test Accuracy: {test_accuracy}")