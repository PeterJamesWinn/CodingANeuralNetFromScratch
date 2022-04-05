# CodingANeuralNetFromScratch
 Coding a neural network from scratch using Numpy in Python.
 This is a toy project to help me understand better the inner workings of a feedforward network vis-a-vis back propagation, the challenges of structuring the code to achieve e.g. batch gradient descent compared to stochastic gradient descent. 
The code was informed by Justin Johnsons Deep Learning for Vision lectures at Michigan, available on Youtube,
and Andrej Karpathy/Justin Johnson/Fei-Fei Li's cs231n lectures at Stanford. 
The initial incarnation of the code was based on the code presented by The Independent Code in the YouTube video "Neural Network from Scratch" but has
had additions (sigmod, relu, binary cross entropy) and restructurings of the implementation presented there, with anticipated future implementations going to introduce many more differences from that start point.




Updates 5th April 2022.

Bug found in the Dense layer. The gradient of the loss with respect to the layer input was being updated after the weights of the layer were updated (probably influenced by this being what The Independent Code YouTube video does). Since the upstream gradient has alread weights already  have been   used to update the weights, using the updated weights to multiply the upstream gradient will lead to approximately the upstream gradient squared being the multiplying factor, if the upstream gradient is very much greater than the weights and the activations arriving from the previous layer. In short this incorrect coding would undoubtly have contributed to exploding gradients during backpropagation. 

A new Binary Cross entropy and associated gradient has been coded with values clipped to avoid log0 and division by 0 issues. 

The code now runs with a sigmoid layer as the last layer prior to binary cross entropy loss, without exploding gradients. Although the results
immediately below for classification are still far from outstanding which might suggest some other fixes are needed e.g. batch normalisation, or further gradient clipping? Or looking at weight initialisation again? 
TestingReluSigmoindAndDenseLayerNetwork_BinaryCrossEntropy.py with 100 epochs gives:
network = [dense_layer(1,6), relu_layer(), dense_layer(6,1), sigmoid_layer()]
# Training parameters
epochs = 150
learning_rate = 0.000003
is sufficient to correctly separate training and testing data sets:
predictions: [ 50  -1  10 -10 -15 -20  20  17] [0.97416525 0.72710304 0.81721557 0.65112824 0.60464477 0.55162105
 0.88397896 0.86654409]
actual values: [ 50  -1  10 -10 -15 -20  20  17] [1 0 1 0 0 0 1 1]
predictions: [ -100  -510     1   100 -1000    -2    29    67] [7.44702508e-02 1.22079040e-06 7.42514457e-01 9.98158243e-01
 2.12523887e-12 7.19184794e-01 9.24869170e-01 9.89398199e-01]
actual values: [ -100  -510     1   100 -1000    -2    29    67] [0 0 1 1 0 0 1 1]

but reruning gives less impressive results: 
predictions: [ 50  -1  10 -10 -15 -20  20  17] [0.04479568 0.13132367 0.10510298 0.15673229 0.17250318 0.17528651
 0.08538916 0.09092224]
actual values: [ 50  -1  10 -10 -15 -20  20  17] [1 0 1 0 0 0 1 1]
predictions: [ -100  -510     1   100 -1000    -2    29    67] [0.99975198 1.         0.12617527 0.01649342 1.         0.13396411
 0.07057831 0.03181102]
actual values: [ -100  -510     1   100 -1000    -2    29    67] [0 0 1 1 0 0 1 1]

but 200 epochs muddles things up again. Getting worse results with the training data, so not overfitting(?).
predictions: [ 50  -1  10 -10 -15 -20  20  17] [0.92958101 0.91172002 0.91588918 0.91568758 0.91842689 0.91260991
 0.91952331 0.91844838]
actual values: [ 50  -1  10 -10 -15 -20  20  17] [1 0 1 0 0 0 1 1]
predictions: [ -100  -510     1   100 -1000    -2    29    67] [1.28759174e-01 5.94808610e-13 9.12491687e-01 9.83068758e-01
 1.25811600e-26 9.11331892e-01 9.22670935e-01 9.53539484e-01]
actual values: [ -100  -510     1   100 -1000    -2    29    67] [0 0 1 1 0 0 1 1]


Network in TestingReluAndDensLayerNetwork_MSE with: network = [dense_layer(1,6), relu_layer(), dense_layer(6,6), relu_layer() , dense_layer(6,6),relu_layer() , dense_layer(6,1)] with   MSE
and 2500 epochs and the network trained on independent variables with values from 1 to 11
fits data
such that: 
predictions: [30 31 32 33 34 35 36 37 38 39 40] [ 95.14747167  98.15454069 101.16160971 104.16867873 107.17574776
 110.18281678 113.1898858  116.19695482 119.20402384 122.21109286
 125.21816188]
actual values: [30 31 32 33 34 35 36 37 38 39 40] [ 95.  98. 101. 104. 107. 110. 113. 116. 119. 122. 125.]
which seems a good result, although conventional linear regression would be perfect.

A simple linear model, i.e. the
network in TestingSimpleDenseLayerNetwork_EvenMoreNodesPerLayer with: network = network = [ dense_layer(1,1)
200 epochs: epochs MSE, same data as above gives:
predictions: [30 31 32 33 34 35 36 37 38 39 40] [ 95.10981058  98.11530539 101.1208002  104.12629501 107.13178982
 110.13728464 113.14277945 116.14827426 119.15376907 122.15926388
 125.1647587 ]
actual values: [30 31 32 33 34 35 36 37 38 39 40] [ 95.  98. 101. 104. 107. 110. 113. 116. 119. 122. 125.]

with 500 epochs gives: 
predictions: [30 31 32 33 34 35 36 37 38 39 40] [ 95.00034405  98.00036126 101.00037848 104.0003957  107.00041291
 110.00043013 113.00044734 116.00046456 119.00048178 122.00049899
 125.00051621]
actual values: [30 31 32 33 34 35 36 37 38 39 40] [ 95.  98. 101. 104. 107. 110. 113. 116. 119. 122. 125.]
which is a good result.

Network in TestingSigmoidAndDenseLayer_MultipleNodesPerLayer: network = [dense_layer(1,6), sigmoid_layer(), dense_layer(6,1)] with MSE and 500 epochs 
fits data such that: 
predictions: [30 31 32 33 34 35 36 37 38 39 40] [9.11932984 9.11946982 9.11958132 9.11967017 9.119741   9.11979746
 9.1198425  9.11987843 9.1199071  9.11992998 9.11994825]
actual values: [30 31 32 33 34 35 36 37 38 39 40] [ 95.  98. 101. 104. 107. 110. 113. 116. 119. 122. 125.]
This is probably to be expected with a sigmoid bottle neck for this sort of fitting, although there are exploding gradient issues that are worth investigating.

 The version of the code as of 28th March 2022. Updates required. 
 1. The  MSE loss function that will take scalar and vector inputs. It currently doesn't evaluate the mean in the loss function, but rather does the division outside, due to my thinking about stochastic gradient descent. But this is one of the first things that needs changing in the next revision. 
 
 2. The binary cross entropy function also needs: i) to take vector imputs ii) checking whether I have the most robust implementation. 

 3. The RunNetwork() function needs refactoring to allow batch processing as well as stochastic gradient descent. 

 4. Batchnormalisation is needed - using the sigmoid function doesn't seem to work well at the moment and I haven't yet found a problem with the coding of the sigmoid, so am considering that batchnormalisation might be all that is needed. 

 5. It would be good to try import jax as np, to see if the numpy functions can be directly replaced with their Jax equivalents, and what execution speed up this might give. 

 6. It seems to make sense to create a network class, with attributes such as learning rate and loss function.
