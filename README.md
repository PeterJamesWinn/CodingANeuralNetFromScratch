# CodingANeuralNetFromScratch
 Coding a neural network from scratch using Numpy in Python.
 This is a toy project to help me understand better the inner workings of a feedforward network vis-a-vis back propagation, the challenges of structuring the code to achieve e.g. batch gradient descent compared to stochastic gradient descent. 
The code was informed by Justin Johnsons Deep Learning for Vision lectures at Michigan, available on Youtube,
and Andrej Karpathy/Justin Johnson/Fei-Fei Li's cs231n lectures at Stanford. 
The initial incarnation of the code was based on the code presented by The Independent Code in the YouTube video "Neural Network from Scratch" but has
had additions (sigmod, relu, binary cross entropy) and restructurings of the implementation presented there, with anticipated future implementations going to introduce many more differences from that start point.

 The version of the code as of 28th March 2022. Updates required. 
 1. The  MSE loss function that will take scalar and vector inputs. It currently doesn't evaluate the mean in the loss function, but rather does the division outside, due to my thinking about stochastic gradient descent. But this is one of the first things that needs changing in the next revision. 
 
 2. The binary cross entropy function also needs: i) to take vector imputs ii) checking whether I have the most robust implementation. 

 3. The RunNetwork() function needs refactoring to allow batch processing as well as stochastic gradient descent. 

 4. Batchnormalisation is needed - using the sigmoid function doesn't seem to work well at the moment and I haven't yet found a problem with the coding of the sigmoid, so am considering that batchnormalisation might be all that is needed. 

 5. It would be good to try import jax as np, to see if the numpy functions can be directly replaced with their Jax equivalents, and what execution speed up this might give. 

 6. It seems to make sense to create a network class, with attributes such as learning rate and loss function.
