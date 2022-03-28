
# This code was informed by Justin Johnsons Deep Learning for Vision lectures at Michigan, available on Youtube,
# Andrej Karpathy/Justin Johnson/Fei-Fei Li's cs231n lectures at Stanford and The Independent Code's video on Youtube, Neural Network from Scratch.

import numpy as np
import tensorflow as tf

def GenerateTrainingData(min,max):
    '''GenerateTrainingData: example of use: DesignMatrix,TrainingValues= GenerateTrainingData(1,11)
    generates data points from 1 to 10 in steps of 1 and uses them to calculate dependent values using 
    the function defined in ModelFuction()'''
    DesignMatrixString=[]  # Generate as string and then convert, below, to array 
    TrainingValues=[]
    for data in range(min,max):
        DesignMatrixString.append(data)
    DesignMatrix=np.asarray(DesignMatrixString)
    TrainingValues=ModelFunction(DesignMatrix)
    return(DesignMatrix,TrainingValues)  

def ModelFunction(DesignMatrix):
    return(5.0+DesignMatrix*3.0)


class layer:  # set generic features to be inherited
  def __init__(self, features_in=1, nodes_in_layer=1):
    self.features_in = feature_in
    self.nodes_in_layer = nodes_in_layer

class dense_layer(): 
  ''' dense_layer: A layer is treated as an independent entity with its own feature vector input, being the original feature vector for 
      the first hidden layer, or the output of the previous layer for every subsequent layer. The output is a vector to be the 
      input for the next layer. The dense layer only performs the matrix multipliation. An activation function must be included as
      a subsequent layer, if desired. '''
  '''Could set up a layer class and inherit self.feature_in and self.nodes_in_layer. 
    Also Forward and reverse pass could be inherited. 
     but seems unnecessary as long as have only one layer type. '''
  def __init__(self, number_of_features=1, nodes_in_layer=1): #default to a 1 node layer, with one input.
    self.number_of_features = number_of_features
    self.nodes_in_layer = nodes_in_layer
    self.weights=np.random.randn(nodes_in_layer, number_of_features)  # weights initialised from a normal distribution.
    self.bias = np.random.randn(nodes_in_layer, 1)
   

  def forward_pass(self,feature_vector): # returns final value of the layer
    self.feature_vector = feature_vector # needed for backward_pass
    return np.dot(self.weights, feature_vector) + self.bias
   

  def backward_pass(self, upstream_gradient, learning_rate): 
    '''dense_layer backward_pass: updates the weights of the layer and the upstreamstream gradient for the next layer 
       (i.e. downstream gradient of this layer - see Justin Johnson's Deep Learning for Vision lecture number 6 for dicussion).
       Currently hardcoded to run gradient descent. Other options to follow!'''
    dL_dw = np.dot(upstream_gradient, np.transpose(self.feature_vector)) # loss function with respect to weights
    #print("upstream_gradient, learning_rate", upstream_gradient, learning_rate)
    #print("dL_dw",dL_dw )
    #print("weights, rate, dL/dw: ", self.weights, learning_rate, dL_dw)
    self.weights += -learning_rate * dL_dw
    self.bias += -learning_rate * upstream_gradient
    dL_dinput = np.transpose(self.weights).dot(upstream_gradient)  # sensitivity of loss function to feature vector of the layer 
    return dL_dinput

class relu_layer():
  def __init__(self):
    pass

  def forward_pass(self, feature_vector):
    self.relu = relu(feature_vector)  # save for backward pass
    return self.relu

  
  def backward_pass(self, upstream_gradient, learning_rate): 
    '''learning rate not needed but is passed because parameter update is embedded in backward pass of other layers.
    Indicates the need to refactor the code!'''
    local_gradient = np.where(self.relu < 0, 0, 1)
    #print("self.relu in backward_pass. upstream gradient: \n {} \n local_gradient:\n {}".format(upstream_gradient, local_gradient))
    dL_dinput = np.array(upstream_gradient) * np.array(local_gradient)  # elementwise multiply
    return dL_dinput

def relu(x):
    ''' relu: relu function'''
    return np.where(x < 0, 0, x)



class sigmoid_layer():
  def __init__(self):
    pass

  def forward_pass(self, feature_vector):
    self.sigmoid = sigmoid(feature_vector)  # save for backward pass
    return self.sigmoid

  
  def backward_pass(self, upstream_gradient, learning_rate): 
    '''learning rate not needed but is passed because parameter update is embedded in backward pass of other layers.
    Indicates the need to refactor the code!'''
    local_gradient = (1 - self.sigmoid)/self.sigmoid
    #print("self.sigmoid in backward_pass. upstream gradient: \n {} \n local_gradient:\n {}".format(upstream_gradient, local_gradient))
    dL_dinput = np.array(upstream_gradient) * np.array(local_gradient)  # elementwise multiply
    return dL_dinput

def sigmoid(x):
    ''' sigmoid: sigmoid function, i.e. 1/(1+ exp(-x)) acting on the input value. If 
  the input is x, then exp(-x) has a large positive x then exp(-x) becomes very small
  and 1/(1+exp(-x)) suffers from rounding errors/numerical instability. Therefore
  need to evaluate as exp(x)/(1+exp(x)) for positive x. the numpy where command is a 
  succint way to code this. '''
    return np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))

# loss functions aka error functions:
def mse(y, y_hat): 
  '''mse: mean squared error loss: 0.5(y_hat - y)squared. y_hat is the estimate from the network, y is the ground truth'''
  return 0.5*np.square(y_hat - y)

def mse_gradient(y, y_hat):
  '''gradient of mse: mean squared error loss: (y_hat - y). y_hat is the estimate from the network, y is the ground truth'''
  return y_hat - y

def binary_cross_entropy(y, y_hat): 
  '''binary cross entropy: y*np.log2(y_hat) + (1-y)*np.log2(1-y_hat) y_hat is the estimate from the network, y is the ground truth'''
  return y*np.log2(y_hat) + (1-y)*np.log2(1-y_hat)

def binary_cross_entropy_gradient(y, y_hat):
  '''gradient of mse: mean squared error loss: (y_hat - y). y_hat is the estimate from the network, y is the ground truth'''
  return (y/y_hat) - (1-y)/(1-y_hat)


def RunNetwork(epochs, X, Y, learning_rate, error_function, error_grad,network):
  for epoch in range(epochs):
    loss = 0
    for x, y in zip(X,Y): # pairing feature vector and dependent variables and then iterating over each pairing. Zip zips the two vectors together into a list of tuples.
      print("next input, x = ", x)
      next_input = x
      #print("epoch", epoch)
      #print("next input: ", next_input)
      for layer in network:  # for each data entry, x, y, from the previous for command, we iterate through the whole network with this loop.
        next_input = layer.forward_pass(next_input)  # output of one layer is to be the input of the next
        #next_input is at this point y_hat, the predicted valuea
      loss += 1/len(Y)*error_function(y,next_input)  # this line is in the loop for all data. Division by len(Y) is because MSE function doesn't actually calculate mean. 
      grad=1/len(Y)*error_grad(y,next_input)  
      for layer in reversed(network):
        grad = layer.backward_pass(grad, learning_rate)  # weights updated on a per data pair basis, i.e. stochastic gradient descent.

    # loss /= len(Y)
    #print("epoch {} of {},  error = {}".format(epoch + 1, epochs, loss))
  for layer in network:
    print("layer: ", layer)
    try:
      print("bias: {} \n weight {} \n".format(layer.bias,layer.weights))
    except:
      pass
  return

def PredictWithNetwork(X, network):
  #print("X arriving in PredictWithNetwork", X)
  results=[]
  for x in X:
    next_input = x
    #print("predicting for x=", x)
    for layer in network: 
      next_input = layer.forward_pass(next_input)
    #print("next prediction", str(next_input))  
    results.append(next_input)  
    #print("results", results)
    #print("x, y", x, next_input)
  return ( np.asarray(results).reshape(-1))