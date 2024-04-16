import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv('data/heart.csv')

# Splitting Features and Target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# Split Data into Training and Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
# print(X.shape, X_train.shape, X_test.shape)

# Model Training
model = LogisticRegression()
model.fit(X_train, Y_train)

# Accuracy Score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Training Accuracy Score: ",training_data_accuracy)

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Testing Accuracy Score: ",test_data_accuracy)

# Build Predictive System
input_data = () # put values into list for test
input_data_np = np.asarray(input_data)
input_data_reshaped = input_data_np.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

if (prediction[0] == 1):
    print("I predict the input person will have heart disease.")
else:
    print("I predict the input person will not have heart disease.")
