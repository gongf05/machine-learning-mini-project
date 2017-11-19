from mxnet import gluon
from mxnet import autograd
from mxnet import ndarray as nd
import pandas as pd
import numpy as np

# load data using pandas
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# concate all dataset except index and label column
# sample input:
#      Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
#      1            5.1           3.5            1.4           0.2      Iris-setosa
all_X = pd.concat((train.loc[:, 'SepalLengthCm':'PetalWidthCm'],
        test.loc[:,'SepalLengthCm':'PetalWidthCm']))


# preprocessing dataseti - scale the data to avoid skewness
numeric_feas = all_X.dtypes[all_X.dtypes != "object"].index
all_X[numeric_feas] = all_X[numeric_feas].apply(
            lambda x: (x - x.mean()) / (x.std()))

# transform labels from string into numerical ones
all_X = pd.get_dummies(all_X, dummy_na=True)

# convert format into matrix which will be fed into ndarray and mxnet
num_train = train.shape[0]

X_train = all_X[:num_train].as_matrix()
X_test = all_X[num_train:].as_matrix()
y_train = train.Species.as_matrix()
y_test = test.Species.as_matrix()

# load dataset into ndarray
X_train = nd.array(X_train)
y_train = nd.array(y_train)
y_train.reshape((num_train, 1))

# test data
num_test = test.shape[0]
X_test = nd.array(X_test)
y_test = nd.array(y_test)
y_test.reshape((num_test, 1))

# define loss function
square_loss = gluon.loss.L2Loss()

# define basic linear regression model here
net = gluon.nn.Sequential()
with net.name_scope():
    net.add(gluon.nn.Dense(1))
net.initialize()

# define data iter
batch_size = 10
dataset = gluon.data.ArrayDataset(X_train, y_train)
data_iter = gluon.data.DataLoader(dataset, batch_size, shuffle=True)

# define trainer instance
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})
net.collect_params().initialize(force_reinit=True)

# train the model!
epochs = 10
for e in range(epochs):
    total_loss = 0
    total_test_loss = 0
    total_sample = 0
    for data, label in data_iter:
        with autograd.record():
            output = net(data)
            loss = square_loss(output, y_train)
        loss.backward()
        total_sample += batch_size
        trainer.step(batch_size)
        # model training error
        total_loss += nd.sum(loss).asscalar()
    print("-----------------------------------------")
    print("Epoch %d, average training loss: %f" % (e, total_loss/total_sample))
    # testing error
    if X_test is not None:
        test_output = net(X_test)
        test_loss = square_loss(test_output, y_test)
        total_test_loss += nd.sum(test_loss).asscalar()
    print("Epoch %d, average testing loss: %f" % (e, total_test_loss/total_sample))


