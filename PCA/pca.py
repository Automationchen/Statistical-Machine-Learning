import numpy as np
import matplotlib.pyplot as plt

data=np.loadtxt('iris.txt',delimiter=',').reshape(-1,5)

att=data[:,0:4]
label=data[:,4]
mu=np.mean(att,axis=0);
std=np.std(att,axis=0)
normatt=att-mu

for i in range(4):
    normatt[:,i]=normatt[:,i]/std[i]

C=normatt.T.dot(normatt)/len(data)
ew,ev=np.linalg.eig(C)

#projection array
a=np.zeros((4,len(data)))

for i in range(4):
    a[i]=ev.T[i].dot(normatt.T)


# sum of all the projection
a=a**2
sum=np.sum(a) 

p=[np.sum(a[0:i]) for i in range(5)]/sum #proportion
D=min(np.where(p>0.95)[0]);print(D)# find the least order

plt.figure()
d=[0,1,2,3,4]
plt.plot(d,p,'-')
plt.savefig('pca.png')
plt.show()