import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



data = pd.read_csv('650_4000.csv', header=None)

training_set, test_set = train_test_split(data, test_size=0.3, random_state=0)

x_train = training_set.iloc[:,0:3474]
y_train = training_set.iloc[:,3474]
x_test = test_set.iloc[:,0:3474]
y_test = test_set.iloc[:,3474]

# !pip install xgboost

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print(y_test_encoded.shape)


dtrain = xgb.DMatrix(x_train, label=y_train_encoded)
dtest = xgb.DMatrix(x_test, label=y_test_encoded)

params = {
    'objective': 'multi:softmax', # or 'multi:softmax' for multi-class classification
    'num_class': 5,
    'max_depth': 3,
    'learning_rate': 0.1,
    'eval_metric': 'mlogloss' # or 'mlogloss' for multi-class classification
}

# Train the model
num_round = 100
bst = xgb.train(params, dtrain, num_round)

# Make predictions 
test_predictions = bst.predict(dtest)
test_predictions = [round(value) for value in test_predictions]

# Compute accuracy
test_accuracy = accuracy_score(y_test, test_predictions)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')