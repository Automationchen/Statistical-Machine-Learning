import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('ldaData.txt').reshape(-1,2)
c1=data[0:50]
c2=data[50:93]
c3=data[93:137]

def gaussian(x,mu,cov):
    return np.exp(-0.5*(x-mu).dot(np.linalg.inv(cov).dot((x-mu).T))) \
            /2/np.pi/np.sqrt(np.linalg.det(cov))

mu1=sum(c1)/len(c1)
mu2=sum(c2)/len(c2)
mu3=sum(c3)/len(c3)

mu=np.sum(data,axis=0)/len(data);cov=(data-mu).T.dot(data-mu)/136
#prior probability
p_c1=50/137
p_c2=43/137
p_c3=44/137 
#class use LDA classifier and missclassified samples
k1=[]
k2=[]
k3=[]
miss=0 

for i in range(137):
    y1=np.log(gaussian(data[i],mu1,cov)*p_c1) #posterior probability
    y2=np.log(gaussian(data[i],mu2,cov)*p_c2)
    y3=np.log(gaussian(data[i],mu3,cov)*p_c3)
    if(y1>y2 and y1>y3):k1=np.append(k1,data[i]).reshape(-1,2)
    if(y2>y3 and y2>y1):k2=np.append(k2,data[i]).reshape(-1,2)
    if(y3>y2 and y3>y1):k3=np.append(k3,data[i]).reshape(-1,2)
plt.subplot(211)
plt.plot(c1[:,0],c1[:,1],'2',label='C1');plt.plot(c2[:,0],c2[:,1],'.',label='C2')
plt.plot(c3[:,0],c3[:,1],'4',label='C3');plt.legend()
plt.subplot(212)
plt.plot(k1[:,0],k1[:,1],'2',label='LDA:C1');plt.plot(k2[:,0],k2[:,1],'.',label='LDA:C2')
plt.plot(k3[:,0],k3[:,1],'4',label='LDA:C3');plt.legend()
plt.savefig('LDA.png');plt.show()

