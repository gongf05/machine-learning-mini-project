import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# read csv data using pandas
admissions = pd.read_csv('binary.csv')

# (1) pre-processing data

# Make dummy variables for rank
data = pd.concat([admissions, pd.get_dummies(admissions['rank'], prefix='rank')], axis=1)
data = data.drop('rank', axis=1)

# Standarize features
for field in ['gre', 'gpa']:
        mean, std = data[field].mean(), data[field].std()
        data.loc[:,field] = (data[field]-mean)/std
                
# Split off random 10% of the data for testing
#np.random.seed(42)
sample = np.random.choice(data.index, size=int(len(data)*0.9), replace=False)
data, test_data = data.ix[sample], data.drop(sample)

# Split into features and targets
features, targets = data.drop('admit', axis=1), data['admit']
features_test, targets_test = test_data.drop('admit', axis=1), test_data['admit']

# (2) 
# Use to same seed to make debugging easier
np.random.seed(42)

n_records, n_features = features.shape
last_loss = None

# Initialize weights
weights = np.random.normal(scale=1 / n_features**.5, size=n_features)

# Neural Network hyperparameters
epochs = 1000
learnrate = 0.8

for e in range(epochs):
    del_w = np.zeros(weights.shape)

    for x, y in zip(features.values, targets):
        # loop through all records, x is input, y is target
        output = sigmoid(np.dot(x, weights))
        # calculate output error
        error = y - output
        # calculate update on weights
        del_w += error * output * ( 1 - output ) * x
    # update weights
    weights += del_w

    # Printing out the mean square error on the training set
    if e % (epochs / 10) == 0:            
        out = sigmoid(np.dot(features, weights))
        loss = np.mean((out - targets) ** 2)
        if last_loss and last_loss < loss:
            print("Train loss: ", loss, "  WARNING - Loss Increasing")
        else:
            print("Train loss: ", loss)
        last_loss = loss

# Calculate accuracy on test data
tes_out = sigmoid(np.dot(features_test, weights))
predictions = tes_out > 0.5
accuracy = np.mean(predictions == targets_test)
print("Prediction accuracy: {:.3f}".format(accuracy))
