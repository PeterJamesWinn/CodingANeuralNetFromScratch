# CodingANeuralNetFromScratch
 Coding a neural network from scratch using Numpy in Python.
 This is a toy project to help me understand better the inner workings of a feedforward network vis-a-vis back propagation, the challenges of structuring the code to achieve e.g. batch gradient descent compared to stochastic gradient descent. 
The code was informed by Justin Johnson's Deep Learning for Vision lectures at Michigan, available on Youtube,
and Andrej Karpathy/Justin Johnson/Fei-Fei Li's cs231n lectures at Stanford. 
The initial incarnation of the code was based on the code presented by The Independent Code in the YouTube video "Neural Network from Scratch" but has
had additions (sigmoid, relu, binary cross entropy) and restructurings of the implementation presented there, with anticipated future implementations going to introduce many more differences from that start point.

Upadate 20th March 2023
Tidying code a little to make more easily read. 
Implementing better initialisation of weights and biases to see effect on 
learning - this implementation is fairly rigid, hard coding division by 
square root on number of input layers, and setting gains for weights
in layer preceding different non-linearity types. 
Implemented a batch update to the network, although not vectorised.
E.g. 
TestingReluAndDenseLayerNetwork_MSE.py 
runs better: 
DesignMatrix, TrainingValues = GenerateTrainingData(1,10) 
epochs = 4000
learning_rate = 0.0002
# define the error function
error_function = mse
error_grad = mse_gradient
network = [dense_layer(1,6, "relu"), relu_layer(), dense_layer(6,6, "relu"), \
          relu_layer() , dense_layer(6,6, "relu"), relu_layer()\
             , dense_layer(6,1)]
gives almost perfect results:
e.g. first try
predictions: [[30 31 32 33 34 35 36 37 38 39 40]] [ 95.  98. 101. 104. 107. 110. 113. 116. 119. 122. 125.]
actual values: [[30 31 32 33 34 35 36 37 38 39 40]] [[ 95.  98. 101. 104. 107. 110. 113. 116. 119. 122. 125.]]
MSE:  2.1002632740604218e-26
repeated training:
predictions: [[30 31 32 33 34 35 36 37 38 39 40]] [ 94.99905066  97.99900471 100.99895876 103.99891281 106.99886686
 109.99882091 112.99877496 115.99872901 118.99868306 121.99863711
 124.99859116]
actual values: [[30 31 32 33 34 35 36 37 38 39 40]] [[ 95.  98. 101. 104. 107. 110. 113. 116. 119. 122. 125.]]
MSE:  1.5525022321920033e-05

Updates 13th April 2022
Created simple functions to generate a data set with 3D feature vector.
Tested network regression problem with 3D input feature vector successfully (TestingReluAndDenseLayerNetwork_MSE_3DFeatureVectors.py). 
Fixed an error in the relu back propagation/gradient function.
Tested current code on a simple one 1 feature vector classification problem:
TestingReluAndDenseLayerNetwork_BinaryCrossEntropy.py, which functions but needs very
small learning rate else we get vanishing gradient problems. 

Next changes: Full batch and mini batch, batch normalisation. Network class. jax.

Testing with 3D feature vector, TestingReluAndDenseLayerNetwork_MSE_3DFeatureVectors.py, 20 epochs of learning shows a functional network that would converge to the correct answer if run for more epochs. 
network = [dense_layer(3,6), relu_layer(),  dense_layer(6,1)]
predictions on testing data : [[40 40 40]
 [41 41 41]
 [42 42 42]
 [43 43 43]
 [44 44 44]
 [45 45 45]
 [46 46 46]
 [47 47 47]
 [48 48 48]
 [49 49 49]] [522.77912993 535.7166879  548.65424586 561.59180382 574.52936178
 587.46691974 600.4044777  613.34203566 626.27959363 639.21715159]
actual values: [[40 40 40]
 [41 41 41]
 [42 42 42]
 [43 43 43]
 [44 44 44]
 [45 45 45]
 [46 46 46]
 [47 47 47]
 [48 48 48]
 [49 49 49]] [[525. 538. 551. 564. 577. 590. 603. 616. 629. 642.]]
predictions on training data: [[1 1 1]
 [2 2 2]
 [3 3 3]
 [4 4 4]
 [5 5 5]
 [6 6 6]
 [7 7 7]
 [8 8 8]
 [9 9 9]] [ 18.21436943  31.1519274   44.08948536  57.02704332  69.96460128
  82.90215924  95.8397172  108.77727516 121.71483313]
actual values: [[1 1 1]
 [2 2 2]
 [3 3 3]
 [4 4 4]
 [5 5 5]
 [6 6 6]
 [7 7 7]
 [8 8 8]
 [9 9 9]] [[ 18.  31.  44.  57.  70.  83.  96. 109. 122.]]



TestingReluAndDenseLayerNetwork_BinaryCrossEntropy.py
network = [dense_layer(1,6), relu_layer(), dense_layer(6,1), sigmoid_layer()]
changed input script to provide data as 2D ndarray instead of matrix
epochs = 20
learning_rate = 0.00003
error_function=binary_cross_entropy  
error_grad=binary_cross_entropy_gradient
Predictions on training data sufficient to separate:
predictions: [[ 50  -1  10 -10 -15 -20  20  17]] [0.86254532 0.0306707  0.1112121  0.02426122 0.02206736 0.02079674
 0.26343908 0.21171042]
actual values: [[ 50  -1  10 -10 -15 -20  20  17]] [[1 0 1 0 0 0 1 1]]
[1 0 1 0 0 0 1 1] [0.86254532 0.0306707  0.1112121  0.02426122 0.02206736 0.02079674
 0.26343908 0.21171042]
 On test data poor prediction performance but network code works:
 predictions: [[ -100  -510     1   100 -1000    -2    29    67]] [0.03583853 0.94063181 0.03453483 0.99865659 0.99995457 0.02889898
 0.45791461 0.96952676]
actual values: [[ -100  -510     1   100 -1000    -2    29    67]] [[0 0 1 1 0 0 1 1]]
test data still gives poor results:
predictions: [[ -100  -510     1   100 -1000    -2    29    67]] [3.00175808e-01 4.72791285e-07 7.20504626e-01 3.16857021e-01
 7.98121048e-14 7.49263265e-01 4.18255481e-01 3.12605686e-01]
actual values: [[ -100  -510     1   100 -1000    -2    29    67]] [[0 0 1 1 0 0 1 1]]


200 epochs
learning_rate = 0.000003
gives inverted predictions, i.e. 1s are small 0s are large.
predictions: [[ 50  -1  10 -10 -15 -20  20  17]] [0.31042779 0.73989957 0.63055005 0.8158694  0.83351801 0.81881635
 0.5198817  0.55380864]
actual values: [[ 50  -1  10 -10 -15 -20  20  17]] [[1 0 1 0 0 0 1 1]]

# Training parameters
epochs = 200
learning_rate = 0.000003
network = [dense_layer(1,6), relu_layer(), dense_layer(6,6), relu_layer() , dense_layer(6,6),relu_layer() , dense_layer(6,1), sigmoid_layer()]
good separation of the training data:
predictions: [[ 50  -1  10 -10 -15 -20  20  17]] [0.82919421 0.10419984 0.33355826 0.04747778 0.03463164 0.03816906
 0.37441053 0.36967842]
actual values: [[ 50  -1  10 -10 -15 -20  20  17]] [[1 0 1 0 0 0 1 1]]
gives better results for test data than the smaller network, but still not perfect:
 Predicted Output: [array([[0.3608378]]), array([[0.05702957]]), array([[0.1417472]]), array([[0.97411842]]), array([[0.13153627]]), array([[0.09307061]]), array([[0.52615792]]), array([[0.90689277]])]
predictions: [[ -100  -510     1   100 -1000    -2    29    67]] [0.3608378  0.05702957 0.1417472  0.97411842 0.13153627 0.09307061
 0.52615792 0.90689277]

Updates 7th April 2022

Updated RunNetwork to take a 2D feature vector and presumably higher dimensions. This involved some fiddling around regarding the difference in how numpy treats an array of shape (2, ) compared to (2,1), the former being a 1D array, the latter being 2D, despite them containing the same number of elements. This works for the test routine:
TestingReluAndDenseLayerNetwork_MSE_2DFeatureVectors.py.

started some work on the function RunNetwork_BatchOptimisation but this is in no way functional

Need further test this version, for 3D feature vector, and then for all the previous test cases. 
Need to continue with the code for a full batch gradient descnent rather than stochastic gradient descent, and from there mini batches, batch normalisation, adagrad and adam.
Also need a good tidy up of everything! 


Updates 5th April 2022.

Bug found in the Dense layer. The gradient of the loss with respect to the layer input was being updated after the weights of the layer were updated (probably influenced by this being what The Independent Code YouTube video does). Since the upstream gradient has weights already  have been   used to update the weights, using the updated weights to multiply the upstream gradient will lead to approximately the upstream gradient squared being the multiplying factor, if the upstream gradient is very much greater than the weights and the activations arriving from the previous layer. In short this incorrect coding would undoubtly have contributed to exploding gradients during backpropagation. 

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

but rerunning gives less impressive results: 
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

 4. Batchnormalisation is needed - using the sigmoid function doesn't seem to work well at the moment and I haven't yet found a problem with the coding of the sigmoid, so am considering that batch normalisation might be all that is needed. 

 5. It would be good to try import jax as np, to see if the numpy functions can be directly replaced with their Jax equivalents, and what execution speed up this might give. 

 6. It seems to make sense to create a network class, with attributes such as learning rate and loss function.
