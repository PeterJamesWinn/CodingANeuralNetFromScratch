
# This code was informed by Justin Johnsons Deep Learning for Vision lectures at Michigan, available on Youtube,
# and Andrej Karpathy/Justin Johnson/Fei-Fei Li's cs231n lectures at Stanford. 
# The initial incarnation of the code was based on the code presented by The Independent Code in the YouTube video "Neural Network from Scratch" but has
# had additions and restructurings of the implementation presented there, with anticipated future implementations going to introduce many more differences
# from that start point.

import numpy as np
import tensorflow as tf

# generating training data
def GenerateTrainingData(min,max):
    '''GenerateTrainingData: example of use: DesignMatrix,TrainingValues= GenerateTrainingData(1,11)
    generates data points from 1 to 10 in steps of 1 and uses them to calculate dependent values using 
    the function defined in ModelFuction()'''
    DesignMatrixString=[]  # Generate as string and then convert, below, to array 
    TrainingValues=[]
    for data in range(min,max):
        DesignMatrixString.append(data)
    DesignMatrix=np.asarray(DesignMatrixString).reshape((1,(max-min)))
    TrainingValues=np.asarray(ModelFunction(DesignMatrix)).reshape((1,(max-min)))
    #DesignMatrix=np.asarray(DesignMatrixString)
    #TrainingValues=np.asarray(ModelFunction(DesignMatrix))
    return(DesignMatrix,TrainingValues)  

def ModelFunction(DesignMatrix):
    return(5.0+DesignMatrix*3.0)

def GenerateTrainingData_2DFeature(min,max):
    '''GenerateTrainingData: example of use: DesignMatrix1, DesignMatrix2, TrainingValues= GenerateTrainingData_2DFeature(1,10)
    generates data points from 1 to 10 in steps of 1 and uses them to calculate dependent values using 
    the function defined in ModelFuction()'''
    DesignMatrixString1=[]  # Generate as string and then convert, below, to array 
    DesignMatrixString2=[]
    TrainingValues=[]
    for data in range(min,max):
        DesignMatrixString1.append(data)
        DesignMatrixString2.append(data)
    DesignMatrix1 = np.asarray(DesignMatrixString1).reshape((1,(max-min)))
    DesignMatrix2 = np.asarray(DesignMatrixString2).reshape((1,(max-min)))
    TrainingValues = np.asarray(ModelFunction2D(DesignMatrix1, DesignMatrix2)).reshape((1,(max-min)))
    #DesignMatrix=np.asarray(DesignMatrixString)
    #TrainingValues=np.asarray(ModelFunction(DesignMatrix))
    return(DesignMatrix1, DesignMatrix2, TrainingValues)  

def ModelFunction2D(DesignMatrix1, DesignMatrix2):
    return(5.0+DesignMatrix1*3.0 + DesignMatrix2*5.0)

def GenerateTrainingData_3DFeature(min,max):
    '''GenerateTrainingData: example of use: DesignMatrix1, DesignMatrix2, DesignMatrix3, TrainingValues= GenerateTrainingData_3DFeature(1,10)
    generates data points from 1 to 10 in steps of 1 and uses them to calculate dependent values using 
    the function defined in ModelFuction()'''
    DesignMatrixString1=[]  # Generate as string and then convert, below, to array 
    DesignMatrixString2=[]
    DesignMatrixString3=[]
    TrainingValues=[]
    for data in range(min,max):
        DesignMatrixString1.append(data)
        DesignMatrixString2.append(data)
        DesignMatrixString3.append(data)
    DesignMatrix1 = np.asarray(DesignMatrixString1).reshape((1,(max-min)))
    DesignMatrix2 = np.asarray(DesignMatrixString2).reshape((1,(max-min)))
    DesignMatrix3 = np.asarray(DesignMatrixString3).reshape((1,(max-min)))
    TrainingValues = np.asarray(ModelFunction3D(DesignMatrix1, DesignMatrix2, DesignMatrix3)).reshape((1,(max-min)))
    return(DesignMatrix1, DesignMatrix2, DesignMatrix3, TrainingValues)  

def ModelFunction3D(DesignMatrix1, DesignMatrix2, DesignMatrix3):
    return(5.0+DesignMatrix1*3.0 + DesignMatrix2*5.0 + DesignMatrix3*5.0)

# network layers
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
    self.feature_vector = np.asarray(feature_vector) # needed for backward_pass
    print("forward pass. Weights:\n  {} \n feature_vector {} \n bias {}" .format(self.weights, feature_vector, self.bias))
    print("forward pass. Shapes: Weights: \n {} \n feature_vector {} \n bias {}" .format(self.weights.shape, feature_vector.shape, self.bias.shape))
    forward_calc = np.dot(self.weights, self.feature_vector) 
    print("forward_calc, value and shape pre addition of bias", forward_calc, forward_calc.shape)
    forward_calc = np.dot(self.weights, self.feature_vector) + self.bias
    print("forward_calc, value and shape post addition of bias", forward_calc, forward_calc.shape)
    # return np.dot(self.weights, self.feature_vector) + self.bias
    return  forward_calc

  def backward_pass(self, upstream_gradient, learning_rate): 
    '''dense_layer backward_pass: updates the weights of the layer and the upstreamstream gradient for the next layer 
       (i.e. downstream gradient of this layer - see Justin Johnson's Deep Learning for Vision lecture number 6 for dicussion).
       Currently hardcoded to run gradient descent. Other options to follow!'''
    dL_dw = np.dot(upstream_gradient, np.transpose(self.feature_vector)) # loss function with respect to weights
    print("Dense: upstream_gradient:\n {} self.feature_vector: \n {} \n".format(upstream_gradient, self.feature_vector))
    print("dL_dw \n",dL_dw,  "\n")
    print("weights, rate, dL/dw: ", self.weights, learning_rate, dL_dw, "\n")
    dL_dinput = np.dot(np.transpose(self.weights), upstream_gradient)  # sensitivity of loss function to feature vector/activations coming into the current layer from the previous layer. This needs to be calculated before the weights update.
    self.weights += -learning_rate * dL_dw
    print("(learning_rate * upstream_gradient),self.bias", (learning_rate * upstream_gradient),self.bias)
    self.bias += -learning_rate * upstream_gradient
    print("np.transpose(self.weights)", np.transpose(self.weights))
    print("upstream_gradient", upstream_gradient)
    return dL_dinput

class relu_layer():
  def __init__(self):
    pass

  def forward_pass(self, feature_vector):
    self.features = feature_vector  # save for backward pass
    return relu(feature_vector)

  
  def backward_pass(self, upstream_gradient, learning_rate): 
    '''learning rate not needed but is passed because parameter update is embedded in backward pass of other layers.
    Indicates the need to refactor the code!'''
    local_gradient = np.where(self.features < 0.0, 0, 1)  # if relu forward received a negative feature return zero or otherwise 1. 
    print("Relu: upstream_gradient:\n {} \n".format(upstream_gradient))
    print(" local gradient, rate", local_gradient, learning_rate)
    dL_dinput = np.array(upstream_gradient) * np.array(local_gradient)  # elementwise multiply
    return dL_dinput

def relu(x):
    ''' relu: relu function'''
    return np.where(x < 0, 0, x) #if x < 0 return zero or otherwise x. 

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
    print("self.sigmoid in backward_pass. upstream gradient: \n {} \n local_gradient:\n {}".format(upstream_gradient, local_gradient))
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
def mse(y, y_hat, number_of_training_examples): 
  '''mse: mean squared error loss: (y_hat - y)squared. y_hat is the estimate from the network, 
  y is the ground truth. This is initially set up for stochastic gradient descent, so only one sample is transferred 
  to the MSE function, which means that the number of training examples also needs to be passed to the function, 
  to allow it to factor these into the calculation. Will rethink if this really makes sense when refactoring code to include
  batch learning. Althought this formulation would also work for batch learning, the current formulation doesn't seem like clean code. '''
  return np.square(y_hat - y)/number_of_training_examples

def mse_gradient(y, y_hat, number_of_training_examples):
  '''gradient of mse: mean squared error loss: (y_hat - y) - ignoring the factor of 2, which will be absorbed into the learning rate. y_hat is the estimate from the network, 
  y is the ground truth. This is initially set up for stochastic gradient descent, so only one sample is transferred 
  to the MSE function, which means that the number of training examples also needs to be passed to the function, 
  to allow it to factor these into the calculation. Will rethink if this really makes sense when refactoring code to include
  batch learning. Althought this formulation would also work for batch learning, the current formulation doesn't seem like clean code. '''
  print("in mse_gradient, y_hat, y, y_hat - y: ", y_hat, y, y_hat-y)
  return (y_hat - y)/number_of_training_examples

def binary_cross_entropy1(y, y_hat, number_of_training_examples): 
  '''binary cross entropy: y*np.log2(y_hat) + (1-y)*np.log2(1-y_hat) 
  y_hat is the estimate from the network, y is the ground truth. This needs revising to allow vector input and
  I need to check if there's a numerically more stable implementation.'''
  return (y*np.log2(y_hat) + (1-y)*np.log2(1-y_hat))/number_of_training_examples

def binary_cross_entropy(y, y_hat, number_of_training_examples): 
  '''binary cross entropy: y*np.log2(y_hat) + (1-y)*np.log2(1-y_hat) 
  y_hat is the estimate from the network, y is the ground truth.'''
  #print("y*np.log2(np.clip((y_hat), 1e-120, None)): ", y*np.log2(np.clip((y_hat), 1e-120, None)), "\n")
  #print("(1-y)*np.log2(np.clip((1-y_hat), 1e-120, None)): ", (1-y)*np.log2(np.clip((1-y_hat), 1e-120, None)),"\n")
  return (y*np.log2(np.clip((y_hat), 1e-120, None)) + (1-y)*np.log2(np.clip((1-y_hat), 1e-120, None)))/number_of_training_examples # clipping to avoid log(0) minus inf. . (y*np.log2(y_hat) + (1-y)*np.log2(1-y_hat))/number_of_training_examples


def binary_cross_entropy_gradient1(y, y_hat, number_of_training_examples):
  '''gradient of mse: mean squared error loss: (y_hat - y). y_hat is the estimate from the network, y is the ground truth'''
  return ((y/y_hat) - (1-y)/(1-y_hat))/number_of_training_examples

def binary_cross_entropy_gradient(y, y_hat, number_of_training_examples):
  '''gradient of mse: mean squared error loss: (y_hat - y). y_hat is the estimate from the network, y is the ground truth'''
  #print("(y/y_hat): ", ((np.clip(y, 1e-120, None))/np.clip(y_hat, 1e-120, None)))
  #print("np.clip((1-y), 1e-120, None)/(np.clip((1-y_hat), 1e-120, None): ", np.clip((1-y), 1e-120, None)/(np.clip((1-y_hat), 1e-120, None)))
  return ((np.clip(y, 1e-120, None)/np.clip(y_hat, 1e-120, None)) - np.clip((1-y), 1e-120, None)/np.clip((1-y_hat), 1e-120, None))/number_of_training_examples # clipping to avoid divide by zero inf. -((y/y_hat) - (1-y)/(1-y_hat))/number_of_training_examples; y/y_hat needs clipping top and bottom otherwise it doesn't tend to 1 as y and y_hat tend to zero.

def binary_cross_entropy_gradient2(y, y_hat, number_of_training_examples):
  '''gradient of mse: mean squared error loss: (y_hat - y). y_hat is the estimate from the network, y is the ground truth'''
  return (np.clip((y-y_hat), 1e-120, None)/np.clip((y_hat-y_hat*y_hat), 1e-120, None)/number_of_training_examples)  # this is incorrect at the moment since y-y_hat can be negative  and if so will get clipped. Otherwise would be a more efficient implementation.


#### functions for training and inference
def RunNetwork(epochs, X, Y, learning_rate, error_function, error_grad,network):
  print("Training Network")
  for epoch in range(epochs):
    print("epoch", epoch)
    loss = 0
    for pointer_position in range(0,len(Y)):
      x = np.transpose(np.copy(X[pointer_position:pointer_position+1]))
      y = np.copy(Y[pointer_position:pointer_position+1])
    
    #for x, y in zip(X,Y): # pairing feature vector and dependent variables and then iterating over each pairing. Zip zips the two vectors together into a list of tuples.
      #print("next input, x = ", x)
      print("x, y ", x, y)
      next_input = x      
      print("next input: ", next_input)
      for layer in network:  # for each data entry, x, y, from the previous for command, we iterate through the whole network with this loop.
        print("Forward Pass: next input: ", next_input)
        next_input = layer.forward_pass(next_input)  # output of one layer is to be the input of the next
        #the variable next_input containts at this point y_hat, the predicted value.
      loss += error_function(y,next_input, len(Y))  # this line is in the loop for all data. Division by len(Y) is because the current loss functions doesn't actually calculate mean. 
      grad = error_grad(y,next_input, len(Y))  
      print("Backpropagation")
      for layer in reversed(network):
        print("grad: ", grad )
        grad = layer.backward_pass(grad, learning_rate)  # weights updated on a per data pair basis, i.e. stochastic gradient descent.

    #print("epoch {} of {},  error = {}".format(epoch + 1, epochs, loss))
  '''  for layer in network:
    print("layer: ", layer)
    try:
      print("bias: {} \n weight {} \n".format(layer.bias,layer.weights))
    except:
      pass
  return
  '''

def RunNetwork_BatchOptimisation(epochs, X, Y, learning_rate, error_function, error_grad,network):
  #This currently is a sketch - a copy and paste of RunNetwork with minor modification. Not yet functional
  for epoch in range(epochs):
    print("epoch", epoch)
    loss = 0  
    next_input = X
    print("next input: ", next_input)
    for layer in network:  # for each data entry, x, y, from the previous for command, we iterate through the whole network with this loop.
      next_input = layer.forward_pass(next_input)  # output of one layer is to be the input of the next
      #the variable next_input containts at this point y_hat, the predicted value.
      print("next input: ", next_input)
    print("just about to calculate loss", X,Y)
    loss += error_function(Y, next_input, len(Y))  # this line is in the loop for all data. Division by len(Y) is because the current loss functions doesn't actually calculate mean. 
    grad = error_grad(Y,next_input, len(Y))  
    for layer in reversed(network):
      grad = layer.backward_pass(grad, learning_rate)  # weights updated on a per data pair basis, i.e. stochastic gradient descent.
  return




def PredictWithNetwork(X, network):
  #print("X arriving in PredictWithNetwork", X)
  results=[]
  #for x in X:
  for pointer_position in range(0,len(X[0:,])):
    x = np.transpose(np.copy(X[pointer_position:pointer_position+1]))
      #y = np.copy(Y[pointer_position:pointer_position+1])
    next_input = x
    print("predicting for x=", x)
    for layer in network: 
      next_input = layer.forward_pass(next_input)
    #print("next prediction", str(next_input))  
    results.append(next_input)  
    #print("results", results)
    #print("x, y", x, next_input)
  print('Input Features: {} \n Predicted Output: {} '.format(X, results))
  return ( np.asarray(results).reshape(-1))