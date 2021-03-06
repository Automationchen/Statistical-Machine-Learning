import numpy as np
import matplotlib.pyplot as plt
np.random.seed(12345)
class neuralNetwork :

    # initialization
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        #  nodes number and learning rate
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        self.lr = learningrate
        #initialization of weights:
          # w1 for weight between input and hidden layer
          # w2 for weight between hidden layer and output
        self.w1=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.w2=np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        # Sigmoid function
        self.sigmoid = lambda x: 1/(1+np.exp(-x))

    def setweights(self, w1, w2):
            self.w1 = w1
            self.w2 = w2
    def train(self, train_in, tr_out):
        x = np.array(train_in, ndmin=2).T
        t = np.array(tr_out, ndmin=2).T
        #forward
        a1= np.dot(self.w1,x)
        z = self.sigmoid(a1)
        a2 = np.dot(self.w2,z)
        y = self.sigmoid(a2)
        # backward error
        output_e = t - y
        hidden_e = np.dot(self.w2.T, output_e)
        # update weights
        self.w2 += self.lr*np.dot((output_e*y*(1.0-y)),z.T)
        self.w1 += self.lr*np.dot((hidden_e*z*(1.0-z)),x.T)

    def test(self, test_in):
        x = np.array(test_in, ndmin=2).T
        a1 = np.dot(self.w1, x)
        z = self.sigmoid(a1)
        a2 = np.dot(self.w2, z)
        y = self.sigmoid(a2)
        return y


# traing network
def train_show():
    # import data
    train_out = np.loadtxt('mnist_small_train_out.txt').reshape((-1, 1))
    train_in = np.loadtxt('mnist_small_train_in.txt', delimiter=',')
    test_out = np.loadtxt('mnist_small_test_out.txt').reshape((-1, 1))
    test_in = np.loadtxt('mnist_small_test_in.txt', delimiter=',')
    n1 = len(train_out);n2 = len(test_out)
    # transform training outputs to form N*10:
    # [1 0 0 0 0 0 0 0 0 0 0] represents number 0
    tr_out = np.zeros((n1, 10)) + 0.01
    for i in range(n1):
        tr_out[i, int(train_out[i])] = 0.99

    input_nodes = 784;hidden_nodes =40;output_nodes = 10;lr =0.04
    # traing times
    epochs = 100
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, lr)
    # start training
    hist_acc=np.zeros((epochs)) # record accurancy
    for e in range(epochs):
        #training data using mini-batch
        train_data = np.hstack((train_in,tr_out))
        np.random.shuffle(train_data)
        train_in = train_data[:, 0:784]
        tr_out = train_data[:, 784:]
        for i in range(1000):
            x=train_in[i]*0.99+0.01
            target=tr_out[i]
            n.train(x,target)

        # test model in test data and calculate accurancy
        y_test = n.test(test_in)
        pre_label = np.argmax(y_test, axis=0)
        right_num = 0
        for j in range(n2):
            if pre_label[j] == test_out[j]: right_num += 1

        # break if accurancy is enough
        if right_num>=0.92*n2:break
        hist_acc[e]=right_num/n2
    # save data
    np.savetxt('w2.txt', n.w2);np.savetxt('w1.txt', n.w1)
    np.savetxt('hist_acc',hist_acc)
    # show test evalation
    plt.figure()
    plt.plot(hist_acc)
    plt.xlabel('epochs'); plt.ylabel('accurancy')
    plt.savefig('hist_acc.png')
    plt.show()

if __name__ == "__main__":

     train_show()


