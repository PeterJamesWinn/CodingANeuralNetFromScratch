#
# This is just a toy code project for understanding better the inner
# workings of a feed forward neural network. It very much needs
# refactoring/redesigning to be useful as production code. 
#
# This code was informed by Justin Johnsons Deep Learning for Vision 
# lectures at Michigan, available on Youtube, and
# Andrej Karpathy/Justin Johnson/Fei-Fei Li's cs231n lectures at Stanford. 
# The initial incarnation of the code was based on the code presented by 
# The Independent Code in the YouTube video "Neural Network from Scratch" 
# but has had additions and restructurings of the implementation  
# presented there, with anticipated future implementations going to  
# introduce many more differences from that start point.

# Below are functions for generating datasets with 1D< 2D, and 3D 
# features, followed by classes methods and functions for the neural 
# network.

import numpy as np

# generating training data
def GenerateTrainingData(min,max):
    '''
    GenerateTrainingData: example of use: 
    DesignMatrix,TrainingValues = GenerateTrainingData(1,11) generates 
    data points from 1 to 10 in steps of 1 and uses them to calculate 
    dependent values using the function defined in ModelFuction()
    '''
    DesignMatrixString=[] #Generate as string. Converted below to array 
    TrainingValues=[]
    for data in range(min,max):
        DesignMatrixString.append(data)
    DesignMatrix = np.asarray(DesignMatrixString).reshape((1,(max-min)))
    TrainingValues = \
          np.asarray(ModelFunction(DesignMatrix)).reshape((1,(max-min)))
    #DesignMatrix=np.asarray(DesignMatrixString)
    #TrainingValues=np.asarray(ModelFunction(DesignMatrix))
    return(DesignMatrix,TrainingValues)  

def ModelFunction(DesignMatrix):
    return(5.0+DesignMatrix*3.0)

def GenerateTrainingData_2DFeature(min,max):
    '''
    GenerateTrainingData: example of use: DesignMatrix1, DesignMatrix2, 
    TrainingValues = GenerateTrainingData_2DFeature(1,10)
    generates data points from 1 to 10 in steps of 1 and uses them 
    to calculate dependent values using 
    the function defined in ModelFuction()
    '''
    DesignMatrixString1=[] #Generate as string. Converted below to array 
    DesignMatrixString2=[]
    TrainingValues=[]
    for data in range(min,max):
        DesignMatrixString1.append(data)
        DesignMatrixString2.append(data)
    DesignMatrix1 = np.asarray(DesignMatrixString1).reshape((1,(max-min)))
    DesignMatrix2 = np.asarray(DesignMatrixString2).reshape((1,(max-min)))
    TrainingValues = np.asarray(ModelFunction2D(DesignMatrix1, \
                                    DesignMatrix2)).reshape((1,(max-min)))
    #DesignMatrix=np.asarray(DesignMatrixString)
    #TrainingValues=np.asarray(ModelFunction(DesignMatrix))
    return(DesignMatrix1, DesignMatrix2, TrainingValues)  

def ModelFunction2D(DesignMatrix1, DesignMatrix2):
    return(5.0+DesignMatrix1*3.0 + DesignMatrix2*5.0)

def GenerateTrainingData_3DFeature(min,max):
    '''
    GenerateTrainingData: example of use: DesignMatrix1, 
    DesignMatrix2, DesignMatrix3, TrainingValues = 
    GenerateTrainingData_3DFeature(1,10) generates data points 
    from 1 to 10 in steps of 1 and uses them to calculate dependent 
    values using the function defined in ModelFuction()
    '''
    DesignMatrixString1=[] #Generate as string. Converted below to array 
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
    TrainingValues = np.asarray(ModelFunction3D(DesignMatrix1, \
                     DesignMatrix2, DesignMatrix3)).reshape((1,(max-min)))
    return(DesignMatrix1, DesignMatrix2, DesignMatrix3, TrainingValues)  

def ModelFunction3D(DesignMatrix1, DesignMatrix2, DesignMatrix3):
    return(5.0+DesignMatrix1*3.0 + DesignMatrix2*5.0 + DesignMatrix3*5.0)

# network layers
class dense_layer(): 
  '''Creates a dense layer''' 
  #dense_layer: A layer is treated as an independent entity with its 
  #own feature vector input, being the original feature vector for 
  #the first hidden layer, or the output of the previous layer for 
  #every subsequent layer. The output is a vector to be the 
  #input for the next layer. The dense layer 
  #only performs the matrix multipliation. An activation function 
  #must be included as
  #a subsequent layer, if desired. 
  #Could set up a layer class and inherit self.feature_in and 
  # self.nodes_in_layer. 
  #Also Forward and reverse pass could be inherited. 
  #but seems unnecessary as long as have only one layer type. 
  def __init__(self, number_of_features=1, nodes_in_layer=1, \
              next_layer="dense"):
    '''
    Dense layer. Default initialisation is a one node layer, with one 
    input. Weights assigned from unit Gaussian scaled by 1/square root
    (input nodes) with additional scaling according to next_layer 
    definition.
    '''
    self.number_of_features = number_of_features
    self.nodes_in_layer = nodes_in_layer

    # weights initialised from a normal distribution.
    # scaled by 1/square root of number of inputs, since this improves
    # initial training of deep networks.
    self.weights = np.random.randn(nodes_in_layer, number_of_features) \
                    / np.sqrt(number_of_features) #Kaiming He fan in correction
    if next_layer == "relu":
        self.weights = self.weights * 2 # Kaiming He gain for a relu.
    # gain for sigmoid and linear are 1, so no further coding required at
    # the moment. Tanh is 5/3 but not coded a Tanh yet. 

    #self.bias = np.random.randn(nodes_in_layer, 1)
    self.bias = 0

  def forward_pass(self,feature_vector): # returns final value of the layer
    self.feature_vector = np.asarray(feature_vector)#need for backward_pass
    #print("forward pass. Weights:\n  {} \n feature_vector {}\
    #  \n bias {}" .format(self.weights, feature_vector, self.bias))
    #print("forward pass. Shapes: Weights: \n {} \n feature_vector {} \n \
    # bias {}" .format(self.weights.shape, feature_vector.shape, \
    #  self.bias.shape))
    #forward_calc = np.dot(self.weights, self.feature_vector) 
    #print("forward_calc, value and shape pre addition of bias", \
    # forward_calc, forward_calc.shape)
    forward_calc = np.dot(self.weights, self.feature_vector) + self.bias
    #print("forward_calc, value and shape post addition of bias",\
    #  forward_calc, forward_calc.shape)
    # return np.dot(self.weights, self.feature_vector) + self.bias
    return  forward_calc

  def backward_pass(self, upstream_gradient, learning_rate): 
    '''dense_layer backward_pass: updates the weights of the layer and 
    the upstreamstream gradient for the next layer 
       (i.e. downstream gradient of this layer - see Justin Johnson's 
       Deep Learning for Vision lecture number 6 for dicussion).
       Currently hardcoded to run gradient descent. Other options 
       to follow!'''
    # loss function with respect to weights
    dL_dw = np.dot(upstream_gradient, np.transpose(self.feature_vector)) 
    #print("Dense: upstream_gradient:\n {} self.feature_vector: \n {} \
    # \n".format(upstream_gradient, self.feature_vector))
    #print("dL_dw \n",dL_dw,  "\n")
    #print("weights, rate, dL/dw: ", self.weights, \
    # learning_rate, dL_dw, "\n")
    # sensitivity of loss function to feature vector/activations coming 
    # into the current layer from the previous layer. This needs to be 
    # calculated before the weights update.
    dL_dinput = np.dot(np.transpose(self.weights), upstream_gradient)  
    self.weights += -learning_rate * dL_dw
    #print("(learning_rate * upstream_gradient),self.bias",\
    #  (learning_rate * upstream_gradient),self.bias)
    self.bias += -learning_rate * upstream_gradient
    #print("np.transpose(self.weights)", np.transpose(self.weights))
    #print("upstream_gradient", upstream_gradient)
    return dL_dinput

class relu_layer():
  def __init__(self):
    pass

  def forward_pass(self, feature_vector):
    self.features = feature_vector  # save for backward pass
    return relu(feature_vector)

  def backward_pass(self, upstream_gradient, learning_rate): 
    '''Return the gradient passing through the relu. Takes gradient
    from upstream layer of gradient flow. Learning rate not needed 
    but is passed because parameter update is embedded in backward 
    pass of dense layers, suggesting a need to refactor the code!'''
    local_gradient = np.where(self.features < 0.0, 0, 1) #if relu forward 
                #received a negative feature, return zero, otherwise 1. 
    #print("Relu: upstream_gradient:\n {} \n".format(upstream_gradient))
    #print(" local gradient, rate", local_gradient, learning_rate)

    # elementwise multiply
    dL_dinput = np.array(upstream_gradient) * np.array(local_gradient)
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
    '''learning rate not needed but is passed because parameter update 
    is embedded in backward pass of other layers.
    Indicates the need to refactor the code!'''
    local_gradient = sigmoid_gradient_sig_arg(self.sigmoid)
    #local_gradient = (1 - self.sigmoid)/self.sigmoid
    #print("self.sigmoid in backward_pass. upstream gradient: \n {} \n \
    # local_gradient:\n {}".format(upstream_gradient, local_gradient))
    #elementwise multiply
    dL_dinput = np.array(upstream_gradient) * np.array(local_gradient) 
    return dL_dinput

def sigmoid(x):
    ''' sigmoid(x) returns 1/(1+ exp(-x))'''
    # I've read that if the input is x, For a large negative x, exp(-x) 
    # becomes very large and 1/(1+exp(-x)) potentially 
    # errors/numerical instability. Therefore need to evaluate as 
    # exp(x)/(1+exp(x)) for negative x. In some quick tests that I did, 
    # the latter was unstable for large positive x. I didn't find a -ve
    # value of x where 1/(1+exp(-x)) was unstable, although this doesn't
    # mean that it doesn't happen! 
    # The numpy "where" command is a 
    # succint way to code this. 
    return np.where(x < 0, np.exp(x)/(1 + np.exp(x)), 1/(1 + np.exp(-x)))

def sigmoid_gradient(x):
  ''' sigmoid_gradient(x) returns (1 - sigmoid(x))* sigmoid(x)'''
  return (1 - sigmoid(x))* sigmoid(x)

def sigmoid_gradient_sig_arg(sigmoid_output):
  '''
  Takes sigmoid_ouput = sigmoid(x) as input;
  returns (1 - sigmoid_ouput) * sigmoid_ouput
  '''
  return (1 - sigmoid_output) * sigmoid_output

# loss functions aka error functions:
def mse(y, y_hat, number_of_training_examples): 
  '''mse: mean squared error loss: (y_hat - y)squared. 
  y_hat is the estimate from the network, 
  y is the ground truth. '''
  return np.square(y_hat - y)/number_of_training_examples

def mse_gradient(y, y_hat, number_of_training_examples):
  '''gradient of mse: mean squared error loss: (y_hat - y) - ignoring 
  the factor of 2, which will be absorbed into the learning rate. y_hat 
  is the estimate from the network, 
  y is the ground truth.''' 
  #print("in mse_gradient, y_hat, y, y_hat - y: ", y_hat, y, y_hat-y)
  return (y_hat - y)/number_of_training_examples

def binary_cross_entropy1(y, y_hat, number_of_training_examples): 
  '''binary cross entropy: y*np.log2(y_hat) + (1-y)*np.log2(1-y_hat) 
  y_hat is the estimate from the network, y is the ground truth.''' 
  # This needs revising to allow vector input and  I need to check if 
  # there's a numerically more stable implementation.
  return (y*np.log2(y_hat)\
           + (1-y)*np.log2(1-y_hat))/number_of_training_examples

def binary_cross_entropy(y, y_hat, number_of_training_examples): 
  '''binary cross entropy: y*np.log2(y_hat) + (1-y)*np.log2(1-y_hat) 
  y_hat is the estimate from the network, y is the ground truth.'''
  # clipping to avoid a log(0) minus inf. in the function:
  # (y*np.log2(y_hat) + (1-y)*np.log2(1-y_hat))/number_of_training_examples
  return (y*np.log2(np.clip((y_hat), 1e-120, None)) + \
          (1-y)*np.log2(np.clip((1-y_hat), 1e-120, None)))\
            /number_of_training_examples 


#def binary_cross_entropy_gradient1(y, y_hat, number_of_training_examples):
#  '''gradient of mse: mean squared error loss: (y_hat - y). y_hat 
#  is the estimate from the network, y is the ground truth'''
#  return ((y/y_hat) - (1-y)/(1-y_hat))/number_of_training_examples

def binary_cross_entropy_gradient(y, y_hat, number_of_training_examples):
  '''
  Returns the gradient of the binary cross entropy:
  y*np.log2(y_hat) + (1-y)*np.log2(1-y_hat) 
  y_hat is the estimate from the network, y is the ground truth.'''
  # clipping to avoid divide by zero inf. 
  # -((y/y_hat) - (1-y)/(1-y_hat))/number_of_training_examples; y/y_hat 
  # needs clipping top and bottom otherwise it doesn't tend to 1 as y and 
  # y_hat tend to zero.
  return ((np.clip(y, 1e-120, None)/np.clip(y_hat, 1e-120, None))\
           - np.clip((1-y), 1e-120, None)/np.clip((1-y_hat), 1e-120, None))\
            /number_of_training_examples 

def binary_cross_entropy_gradient2(y, y_hat, number_of_training_examples):
  '''gradient of mse: mean squared error loss: (y_hat - y). y_hat is the 
  estimate from the network, y is the ground truth'''
  # this is incorrect at the moment since y-y_hat can be negative 
  # and if so will get clipped. Otherwise would be a more efficient 
  # implementation.
  return (np.clip((y-y_hat), 1e-120, None)/np.clip((y_hat-y_hat*y_hat),\
          1e-120, None)/number_of_training_examples)  


#### functions for training and inference
def RunNetwork(epochs, X, Y, learning_rate, \
               error_function, error_grad, network):
  '''Run stochastic gradient descent'''
  print("Training Network")
  for epoch in range(epochs):
    print("epoch", epoch)
    loss = 0
    for pointer_position in range(0,len(Y)):
      x = np.transpose(np.copy(X[pointer_position:pointer_position+1]))
      y = np.copy(Y[pointer_position:pointer_position+1])
      next_input = x      
      #print("next input: ", next_input)
      #for each data point, x, y, iterate thru' all layers of the network.
      for layer in network:         
        #print("Forward Pass: next input: ", next_input)       
        # each layer's output is the input of the next
        next_input = layer.forward_pass(next_input) 
        
      # next_input now contains the value output from the final layer.
      y_hat = next_input
      print("prediction, y_hat: ",y_hat )
      loss = error_function(y, y_hat, 1)                          
      print("loss: ", loss)
      grad = error_grad(y, next_input, 1)  
      
      #print("Backpropagation")
      for layer in reversed(network):
        #print("grad: ", grad )
        # weights updated on a per data pair basis, i.e. 
        # stochastic gradient descent.
        grad = layer.backward_pass(grad, learning_rate)  

    #print("epoch {} of {},  error = {}".format(epoch + 1, epochs, loss))

def RunNetwork_BatchOptimisation(epochs, X, Y, learning_rate, error_function,\
                                 error_grad,network, batch_size):
  #This currently is a sketch - a copy and paste of RunNetwork 
  # with minor modification. This is a non-vectorised version. 
  # back propagating average gradient per batch_size.
  # so iterate of batch_size number of entries, calculating grad,
  # before going to backprop.
  for epoch in range(epochs):
    if epoch%10 == 0: print("epoch", epoch)
    loss = 0
    grad = 0
    counter = 0
    for pointer_position in range(0,len(Y)):
      x = np.transpose(np.copy(X[pointer_position:pointer_position+1]))
      y = np.copy(Y[pointer_position:pointer_position+1])
      #print("next input, x = ", x, "counter", counter)
      next_input = x
      for layer in network:  
        next_input = layer.forward_pass(next_input) 
        # the variable next_input containts at this point y_hat, \
        # the predicted value.
      loss += error_function(y, next_input, 1) # below divide by counter
      grad += error_grad(y, next_input, 1) # below divide by counter
      counter += 1
      if counter >= batch_size: 
        # 1. backprop. with previous gradient  
        #print("counter:", counter)
        grad = grad/counter
        for layer in reversed(network):
          grad = layer.backward_pass(grad, learning_rate) 
        # 2. reset for next batch.
        loss = 0
        grad = 0
        counter = 0 
    if counter > 0: # broke loop before achieving full batch
      # complete the partial batch
      print("counter:", counter)
      grad = grad/counter
      for layer in reversed(network):
        grad = layer.backward_pass(grad, learning_rate) 
      # 2. no next batch, on this pass. 
  return

def PredictWithNetwork(X, network):
  '''
  Returns predictions of dependent data Y from independent data X, 
  using network model, network. X is required to be a 2D numpy array,
  even though only contains a 1D list in it. Need to fix this! 
  '''
  #print("X arriving in PredictWithNetwork", X)
  results=[]
  for pointer_position in range(0,len(X[0:,])):
    x = np.transpose(np.copy(X[pointer_position:pointer_position+1]))
    next_input = x
    #print("predicting for x=", x)
    for layer in network: 
      next_input = layer.forward_pass(next_input)
    #print("next prediction", str(next_input))  
    results.append(next_input)  
    #print("results", results)
    #print("x, y", x, next_input)
  print('Predicting: \n Input Features: {} \n Predicted Output: {} \n'.format(X, results))
  return ( np.asarray(results).reshape(-1))