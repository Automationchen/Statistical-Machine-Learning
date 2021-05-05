import  numpy as np
import math
import matplotlib.pyplot as plt
import operator

trainset = np.loadtxt('nonParamtrain.txt', dtype=float)

def distance_1d(trainset,x,K):
    N = len(trainset)
    train = trainset.reshape((-1, 1))
    distances = np.tile(x,(N,1))-train
    distance = np.abs(distances) # norm between trainset and input data
    dist = distance[np.argsort(distance)] # sorted by ascending order
    #print(dist)
    return distance[K-1]

def KNN_estimator(K,N,V):
    pd = K/(N * V)
    return pd

def main():
    N = len(trainset)
    X_plot = np.linspace(-4,8,80)
    fig, ax = plt.subplots()
    K = [2,8,35]
    V0 = distance_1d(trainset, X_plot, K[0])
    V1 = distance_1d(trainset, X_plot, K[1])
    V2 = distance_1d(trainset, X_plot, K[2])
    plt.plot(X_plot, KNN_estimator(K[0], N, V0), 'r')
    plt.plot(X_plot, KNN_estimator(K[1], N, V1), 'g')
    plt.plot(X_plot, KNN_estimator(K[2], N, V2), 'b')
    plt.title('k for red,green and blue are 2; 8 and 35  ')
    #ax.legend(loc='lower right')
    plt.show()
if __name__ == '__main__':
        main()