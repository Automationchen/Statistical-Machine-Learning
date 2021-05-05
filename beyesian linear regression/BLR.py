import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('linRegdata.txt').reshape(150,2)
p=[10,12,18,20,50,150]
dim=13
beta=1/0.0025
lamda=0.000001
alpha=lamda*beta

# X :training input set;y:training output set
# x:predictive point
def mean_variance(x,X,y):

    N=len(X)
    phi=np.array([[x ** i for i in range(dim)]] ).T
    Phi=np.array([[X[j] ** i for i in range(dim)] for j in range(N)]).T
    mu=phi.T.dot(np.linalg.inv(Phi.dot(Phi.T)+lamda*np.eye(dim))).dot(Phi).dot(y.T)
    sigma=1/beta+phi.T.dot(np.linalg.inv(alpha*np.eye(dim)+beta*Phi.dot(Phi.T))).dot(phi)

    return mu,sigma

for j in range(len(p)):

    sup=[231,232,233,234,235,236]
    title=['first 10 points','first 12 points','first 18 points',
           'first 20 points','first 50 points','first 150 points']
    plt.subplot(sup[j])
    mu=[];sigma=[]
    for i in range(150):
        m,s=mean_variance(data[i,0],data[0:p[j],0],data[0:p[j],1])
        mu=np.append(mu,m)
        sigma=np.append(sigma,s)
    sum=mu+sigma;diff=mu-sigma
    plt.plot(data[:,0],mu,'.',label='mean')
    plt.plot(data[:,0],sigma,'.',label='variance')
    plt.fill_between(data[:,0],sum,diff,facecolor='yellow')
    plt.legend()
    plt.title(title[j])
plt.savefig('blr.png');plt.show()