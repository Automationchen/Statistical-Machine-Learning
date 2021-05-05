import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

file1 = np.loadtxt('../nonParamTrain.txt', dtype=float)

def gaussian_probability(x,h):
    N = len(x)
    mu = sum(x)/ N
    eq1 = sum(np.exp(-x ** 2 / (2 * h ** 2)))
    guassian =  eq1/ (N * h * np.sqrt(2 * np.pi))
    return guassian

def gaussian_estimator(x,h):
    N = len(x)
    mu = sum(x)/ N
    eq1 = np.exp(-x ** 2 / (2 * h ** 2))
    guassian =  eq1/ (N * h * np.sqrt(2 * np.pi))
    return guassian

def gaussian(x,b=1):
    return np.exp(-x**2/(2*b**2))/(b*np.sqrt(2*np.pi))

def main():
    # compute the log-likelihood of the data for each class.
    gaussian1 = np.log(gaussian_probability(file1, 0.03))
    gaussian2 = np.log(gaussian_probability(file1, 0.2))
    gaussian3 = np.log(gaussian_probability(file1, 0.8))
    print('gaussian1__________________________\n',
          gaussian1, '\ngaussian2_________________\n',
          gaussian2, '\ngaussian3_________________\n', gaussian3)  # delta = 0.2 is the best

    print('gaussian-----',gaussian(file1,b=1))
    # show different density estimates
    N = len(file1)
    X_plot = np.linspace(-4, 8, 12)
    fig, ax = plt.subplots()

    sum1 = np.zeros(len(X_plot))
    sum2 = np.zeros(len(X_plot))
    sum3 = np.zeros(len(X_plot))
    for i in range(0, N):
        #ax.plot(X_plot, (gaussian_estimator(X_plot - file1[i], 0.03)), 'green', linestyle="dashed")
        sum1 += (gaussian(X_plot - file1[i],0.03))
        sum2 += (gaussian(X_plot - file1[i],0.2))
        sum3 += (gaussian(X_plot - file1[i],0.8))
    ax.fill(X_plot, sum1, '-k',fc='#AAAAFF',alpha=0.3,label='delta is 0.03')
    ax.fill(X_plot, sum2, '-k',fc='green',alpha=0.1,label='delta is 0.2')
    ax.fill(X_plot, sum3, '-k',fc='red',alpha=0.2,label='delta is 0.8') # delta = 0.8 is am besten

    ax.legend(loc='lower right')
    plt.show()

if __name__ == '__main__':
    main()
