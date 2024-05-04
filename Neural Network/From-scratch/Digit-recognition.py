import pandas
import numpy as np
import math
from matplotlib import pyplot as plt
data = pandas.read_csv('Neural Network/From-scratch/train.csv')
data = data.to_numpy()
m,n = data.shape
# TO display images
def showimg(img,label = None):
    plt.xlabel(label)
    img = img.reshape(28,28)
    plt.gray()
    plt.imshow(img)
    plt.show()
def relu(i):
    return np.maximum(0,i)

def softmax(x):
    sum = 0
    for i in range(len(x)):
        x[i] = np.exp(x[i])
        sum += x[i]
    x = x/sum
    return x
def randomweights():
    w1 = np.random.randint(-5,6, size=(10,784))
    w2 = np.random.randint(-5,6, size=(10,10))
    b1 = np.random.randint(-5,6, size=(10))
    b2 = np.random.randint(-5,6, size=(10))
    return [w1,w2,b1,b2]
def one_hot(y):
    one_hot_y = [0,0,0,0,0,0,0,0,0,0]
    one_hot_y[y] = 1
    return one_hot_y
def network_forward(x,weight):
    w1,w2,b1,b2 = weight
    print(x)
    print(w1.shape)
    z1 = np.dot(w1,x) + b1
    za1 = relu(z1)
    z2 = np.dot(w2,za1) + b2
    za2 = softmax(z2)
    # print(za2)
    return z1,za1,z2,za2
def relu_deriv(x):
    for i in range(len(x)):
        if x[i] > 0:
            x[i] = 1
        else:
            x[i] = 0
    return x
def backward_prop(Z1, A1, Z2, A2, W1, W2, x, Y):
    m = len(Y)
    one_hot_Y = one_hot(Y)
    print(one_hot_Y)
    print(A2)
    dZ2 = A2 - one_hot_Y 
    print(dZ2)
    dW2 = 1 / m * np.dot(dZ2,A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = np.dot(W2.T,dZ2) * relu_deriv(Z1)
    dW1 = 1 / m * np.dot(dZ1,x.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2
def gradient_decent(y,X,iter,alpha):
    print("Gradient Decent activated")
    # print(x.shape)
    print(X)
    w1,w2,b1,b2 = randomweights()
    for i in range(iter):
        validation([w1,w2,b1,b2],X,y)
        print("\nIteration: ",i)
        # for j in range(len(y)):
        z1,za1,z2,za2 = network_forward(X,[w1,w2,b1,b2])
        dw1,db1,dw2,db2 = backward_prop(z1,za1,z2,za2,w1,w2,X,y)
        # update weights:
        w1 = w1 - alpha*dw1
        w2 = w2 - alpha*dw2
        b1 = b1 - alpha*db1
        b2 = b2 - alpha*db2
    return w1,w2,b1,b2
def prediction(x,weights):
    _,_,_,Y = network_forward(x,weights)
    for i in range(len(Y)):
        if Y[i] == max(Y):
            return(i)
def validation(weights,x_test,y_test):
    correct = 0
    for i in range(len(y_test)):
        if prediction(x_test[i],weights) == y_test[i]:
            correct += 1
    accuracy = (correct/len(y_test))*100
    print("Accuracy: ",accuracy,"%")
    return accuracy
# def network_backward(z1,za1,z2,za2):   
# Splitting into training and test data
train = data[:100].T
y_train = train[0]
x_train = train[1:]/255
test = data[21000:].T
y_test = test[0]
x_test = test[1:]/255
# print(x_train.shape)
w1,w2,b1,b2 = gradient_decent(y_train,x_train,5,0.5)