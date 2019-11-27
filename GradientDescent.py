import pandas as pd
import numpy as np
from matplotlib import pyplot as plt


def readFile(filename):
    df = pd.read_csv(filename, sep=" ", header=None)
    df1 = pd.DataFrame((x.split(',') for x in df[0]), dtype=np.float)
    return np.array(df1)


def standardize(dataArr):
    avg = np.mean(dataArr, axis=0)
    std = np.std(dataArr, axis=0)
    return (dataArr - avg) / std


def sigmoid(x):
    a = 1/(1+np.exp(-x))
    return a


def train(X, y, nIter, learningRate, lambda_reg):
    [m, n] = np.shape(X)  # size of X
    # initialize parameters
    W = np.zeros(shape=(n, 1), dtype=float)
    dW = np.zeros(shape=(n, 1), dtype=float)
    b = 0
    db = 0

    costs = []
    # iterations for train
    for i in range(nIter):
        # propagation
        yhat = sigmoid(np.dot(X, W)+b)
        reg_term = (lambda_reg/(2*m)) * np.sum(np.dot(W, W.T))
        cost = -(np.sum(y*np.log(yhat)+(1-y)*np.log(1-yhat)))/m + reg_term
        costs.append(cost)
        dy = yhat - y

        # back propagation
        dW = (np.sum(np.dot(X.T, dy), axis=0) + lambda_reg*W)/m
        db = (np.sum(dy))/m

        # update parameters
        W = W - learningRate * dW
        b = b - learningRate * db
    return W, b, costs


def accuracy(W, b, X, y):
    [m, n] = X.shape
    yhat = sigmoid(np.dot(X, W)+b)
    threshold = 0.5
    yhat[np.where(yhat > threshold)] = 1
    yhat[np.where(yhat <= threshold)] = 0
    return 1 - (np.mean(np.abs(yhat - y)))


if __name__ == "__main__":
    # Read data
    data = readFile("spambase.data")

    # divide the data into X and Y,i.e., features and label
    [m, n] = np.shape(data)  # [nb_rows, nb_cols]

    dataX = data[:, np.arange(0, n-1)]
    y = data[:, n-1].reshape(-1, 1)

    # standardize N(0,1) mean=0, standard deviation=1
    X = standardize(dataX)

    W, b, costs = train(X, y, nIter=20, learningRate=0.05, lambda_reg=0)

    acc = accuracy(W, b, X, y)
    print("acc: ", acc)

    # plot the cost to see if it is converging
    # plt.plot(costs)
    # plt.show()
