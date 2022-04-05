
import numpy as np
from NeuralNetInNumpy import *

"""
def binary_cross_entropy(y, y_hat, number_of_training_examples): 
  '''binary cross entropy: y*np.log2(y_hat) + (1-y)*np.log2(1-y_hat) 
  y_hat is the estimate from the network, y is the ground truth. This needs revising to allow vector input and
  I need to check if there's a numerically more stable implementation.'''
  print("y*np.log2(np.clip((y_hat), 1e-120, None)): ", y*np.log2(np.clip((y_hat), 1e-120, None)), "\n")
  print("(1-y)*np.log2(np.clip((1-y_hat), 1e-120, None)): ", (1-y)*np.log2(np.clip((1-y_hat), 1e-120, None)),"\n")
  return (y*np.log2(np.clip((y_hat), 1e-120, None)) + (1-y)*np.log2(np.clip((1-y_hat), 1e-120, None)))/number_of_training_examples # clipping to avoid log(0) minus inf. . (y*np.log2(y_hat) + (1-y)*np.log2(1-y_hat))/number_of_training_examples


def binary_cross_entropy_gradient(y, y_hat, number_of_training_examples):
  '''gradient of mse: mean squared error loss: (y_hat - y). y_hat is the estimate from the network, y is the ground truth'''
  print("(y/y_hat): ", ((np.clip(y, 1e-120, None))/np.clip(y_hat, 1e-120, None)))
  print("np.clip((1-y), 1e-120, None)/(np.clip((1-y_hat), 1e-120, None): ", np.clip((1-y), 1e-120, None)/(np.clip((1-y_hat), 1e-120, None)))
  return ((np.clip(y, 1e-120, None)/np.clip(y_hat, 1e-120, None)) - np.clip((1-y), 1e-120, None)/np.clip((1-y_hat), 1e-120, None))/number_of_training_examples # clipping to avoid divide by zero inf. ((y/y_hat) - (1-y)/(1-y_hat))/number_of_training_examples; y/y_hat needs clipping top and bottom otherwise it doesn't tend to 1 as y and y_hat tend to zero.


def binary_cross_entropy_gradient2(y, y_hat, number_of_training_examples):
  '''gradient of mse: mean squared error loss: (y_hat - y). y_hat is the estimate from the network, y is the ground truth'''
  return(np.clip((y-y_hat), 1e-120, None)/np.clip((y_hat-y_hat*y_hat))/number_of_training_examples)  # this is incorrect at the moment since y-y_hat can be negative  and if so will get clipped. Otherwise would be a more efficient implementation.
"""





# 0. effect of adding clipping - removes infinities.
# 1. training/test data
# 2. Example output from binary_cross_entropy and binary_cross_entropy_gradient
# 3. testing in vector and iterative modes


y1 = [1,1,1,1]  # must necessarily be 1 or 0 for binary cross entropy known categories
y_hat1 = [1,1,1,1]
y_hat2 = [0,0,0,0]
y_hat3 = [1,1,0,0]
y_hat11 = [0.99,0.99,0.99,0.99]
y_hat12 = [0,0,0,0]
y_hat13 = [0.99,0.99,0,0]
y_hat21 = [0.5,0.5,0.75,0.75]
y_hat22 = [0.01, 0.01, 0.01, 0.01]
y2 = [0,0,0,0]
y3 = [1,1,0,0]
y4 = [0,0,1,1]


number_of_training_examples = 4

for a in [y1,  y2, y3, y4]:
#for a in [y1,  y2]:
    for b in [y_hat1, y_hat2, y_hat3, y_hat11, y_hat12, y_hat13, y_hat21, y_hat22]:
    #for b in [y_hat1, y_hat2]:
        y, y_hat = np.array(a), np.array(b)
        print("\n y, y_hat: ", y, y_hat)
        print("binary_cross_entropy: ", binary_cross_entropy(y,y_hat, 4))
        print("binary_cross_entropy_gradient: ", binary_cross_entropy_gradient(y,y_hat, 4),"\n \n"  )
        #print("binary_cross_entropy_gradient2: ", binary_cross_entropy_gradient2(y,y_hat, 4),"\n \n" )



