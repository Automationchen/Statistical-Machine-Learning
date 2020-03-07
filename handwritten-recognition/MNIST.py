import numpy as np
import matplotlib.pyplot as plt

train_out=np.loadtxt('mnist_small_train_out.txt').reshape((-1,1))
train_in=np.loadtxt('mnist_small_train_in.txt',delimiter=',')
test_out=np.loadtxt('mnist_small_test_out.txt').reshape((-1,1))
test_in=np.loadtxt('mnist_small_test_in.txt',delimiter=',')
n1=len(train_out);n2=len(test_out)

# transform outputs to form N*10:
# [1 0 0 0 0 0 0 0 0 0 0] represents number 0
tr_out=np.zeros((n1,10))+0.01
for i in range(n1):
    tr_out[i,int(train_out[i])]=0.99

class neuralNetwork :

    # initialization
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # input nodes
        self.inodes = inputnodes
        # hidden layer nodes
        self.hnodes = hiddennodes
        # out nodes
        self.onodes = outputnodes
        # learning rate
        self.lr = learningrate
        #initialization of weights:
          # w1 for weight between input and hidden layer
          # w2 for weight between hidden layer and output
        self.w1=np.random.normal(0.0,pow(self.hnodes,-0.5),(self.hnodes,self.inodes))
        self.w2=np.random.normal(0.0,pow(self.onodes,-0.5),(self.onodes,self.hnodes))
        # Sigmoid function
        self.sigmoid = lambda x: 1/(1+np.exp(-x))
    def setweights(self, w1, w2):
        self.w1 =w1
        self.w2 = w2

    def train(self, train_in, tr_out):
        inputs = np.array(train_in, ndmin=2).T
        label = np.array(tr_out, ndmin=2).T
        #forward
        hidden_inputs = np.dot(self.w1, inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)
        final_inputs = np.dot(self.w2, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)
        # backward
        output_errors = label - final_outputs
        hidden_errors = np.dot(self.w2.T, output_errors)
        # update weights
        self.w2 += self.lr * np.dot((output_errors * final_outputs * (1.0 - final_outputs)), np.transpose(hidden_outputs))
        self.w1 += self.lr * np.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), np.transpose(inputs))

    def test(self, test_in):
        inputs = np.array(test_in, ndmin=2).T
        hidden_inputs = np.dot(self.w1, inputs)
        hidden_outputs = self.sigmoid(hidden_inputs)
        final_inputs = np.dot(self.w2, hidden_outputs)
        final_outputs = self.sigmoid(final_inputs)
        return final_outputs
# traing network
def train_show():
    input_nodes = 784;hidden_nodes =200;output_nodes = 10;lr = 0.08
    # traing times
    epochs = 50
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, lr)
    # start training
    hist_acc=np.zeros((epochs)) # record accurancy
    for e in range(epochs):
        for i in range(n1):
            inputs=train_in[i]*0.99+0.01
            label=tr_out[i]
            n.train(inputs, label)
        outputs = n.test(test_in)
        pre_label = np.argmax(outputs, axis=0)
        right_num = 0
        for j in range(n2):
            if pre_label[j] == test_out[j]: right_num += 1

        if right_num>0.92*n2:break

        hist_acc[e]=right_num/n2

    np.savetxt('w2.txt', n.w2);np.savetxt('w1.txt', n.w1)
    np.savetxt('hist_acc',hist_acc)
    plt.figure()
    plt.plot(hist_acc)
    plt.xlabel('epochs'); plt.ylabel('accurancy')
    plt.savefig('hist_acc.png')
    plt.show()
def test_acc():
    input_nodes = 784
    hidden_nodes =200
    output_nodes = 10
    lr= 0.08
    n =neuralNetwork(input_nodes, hidden_nodes, output_nodes, lr)
    w1=np.loadtxt('w1.txt')
    w2=np.loadtxt('w2.txt')
    n.setweights(w1, w2)
    outputs=n.test(test_in)
    pre_label=np.argmax(outputs,axis=0)
    acc=0
    for i in range(n2):
        if pre_label[i]==test_out[i]:acc+=1
    print('accurancy is:',acc/n2)

if __name__ == "__main__":
    train_show()
    #test_acc()

