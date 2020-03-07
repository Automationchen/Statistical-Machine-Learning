# hand-written digits recognition

The dataset mnist_small, divided into four files. The mnist dataset is widely used as benchmark for classification algorithms. It contains 28x28 images of handwritten digits (pairs <input, output>
correspond to <pixels, digit>).


Use squared loss function $L =\frac{1}{2}(y − t)^2$ ,because it is easy to combine with learing effect and to get derivation.\\

We choose a MLP with 2 layers,i.e. a hidden layer.If one choose a MLP with more layers ,the gradient of first few weight will disappear.So the backpropagation will not work any more.And with 2 layers one can esimate
arbitrary functions.\\
After comparsion of defferent learning effects,we select 40 neurons with Sigmoid function.If neurons number is too large ,it will easily overfitting.
Set learning rate with 0.04
