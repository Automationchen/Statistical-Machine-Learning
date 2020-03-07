import numpy as np
import matplotlib.pyplot as plt

train_out=np.loadtxt('mnist_small_train_out.txt').reshape((-1,1))
train_in=np.loadtxt('mnist_small_train_in.txt',delimiter=',')
test_out=np.loadtxt('mnist_small_test_out.txt').reshape((-1,1))
test_in=np.loadtxt('mnist_small_test_in.txt',delimiter=',')
n1=len(train_out);n2=len(test_out)
tr_out=np.zeros((n1,10))+0.01
for i in range(n1):
    tr_out[i,int(train_out[i])]=0.99

for i in range(3):
    train_data=np.hstack((train_in,tr_out))
    np.random.shuffle(train_data)
    train_in=train_data[:,0:784]
    tr_out=train_data[:,784:]
    print(np.shape(train_in))