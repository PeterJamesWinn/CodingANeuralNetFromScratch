

from NeuralNetInNumpy import *

def RunNetwork2(epochs, X, Y, learning_rate, error_function, error_grad,network):
  for epoch in range(epochs):
    loss = 0
    for x, y in zip(X,Y): # pairing feature vector and dependent variables and then iterating over each pairing. Zip zips the two vectors together into a list of tuples.
      print("next input, x = ", x)
      next_input = x
      #print("epoch", epoch)
      #print("next input: ", next_input)
      for layer in network:  # for each data entry, x, y, from the previous for command, we iterate through the whole network with this loop.
        next_input = layer.forward_pass(next_input)  # output of one layer is to be the input of the next
        print("next layer output is: ", next_input)
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



DesignMatrix,TrainingValues= np.matrix([50, -1, 10,-10,-15, -20,   20, 17]), np.matrix([ 1, 0, 1, 0, 0, 0,  1, 1])
#DesignMatrix,TrainingValues= GenerateTrainingData(1,11)
X=np.transpose(DesignMatrix)  # make column vector inputs
Y=np.transpose(TrainingValues)
#X = DesignMatrix
#Y = TrainingValues
print(X, "\n",Y)
network = [dense_layer(1,6), relu_layer(), dense_layer(6,6), relu_layer() , dense_layer(6,6),relu_layer() , dense_layer(6,1)]
#network = [dense_layer(1,6), relu_layer(), dense_layer(6,1), relu_layer()]
# relu doesn't have nodes to define. Data is passed in relu_layer.forward(data) call, when the network is evaluated. 
#X -= np.mean(X, axis=0)
factor = np.std(X)
print("factor", factor)
X = X/factor
print(X,Y)
epochs = 500
learning_rate = 0.00003
# define the error function
error_function=binary_cross_entropy  
error_grad=binary_cross_entropy_gradient
RunNetwork2(epochs, X, Y, learning_rate, error_function, error_grad, network)
#DesignMatrix,TestingValues= np.array([-100,-510, 1, 100, -1000, -2,  29, 67]), np.array([0, 0, 1, 1, 0, 0,  1, 1])
DesignMatrix,TestingValues= np.array([50, -1, 10,-10,-15, -20,   20, 17]), np.array([ 1, 0, 1, 0, 0, 0,  1, 1])
X=np.transpose(DesignMatrix)  # make column vector inputs
#Y=np.transpose(TestingValues)
#X = DesignMatrix
#Y = TestingValues
X= X/factor
#DesignMatrix,TestingValues= GenerateTrainingData(30,41)
Predicted_Values=PredictWithNetwork(X, network)
print("actual values:", DesignMatrix,TestingValues )

import matplotlib.pyplot as plt
plt.title('Predicted values compared to actual values - training set.' )
plt.xlabel('ground truth')
plt.ylabel("prediction")
plt.scatter(TestingValues, Predicted_Values)
plt.show()

DesignMatrix,TestingValues= np.array([-100,-510, 1, 100, -1000, -2,  29, 67]), np.array([0, 0, 1, 1, 0, 0,  1, 1])
X=np.transpose(DesignMatrix)  # make column vector inputs
Y=np.transpose(TestingValues)
X= X/factor
#DesignMatrix,TestingValues= GenerateTrainingData(30,41)
Predicted_Values=PredictWithNetwork(X, network)
print("predictions:", DesignMatrix, Predicted_Values)
print("actual values:", DesignMatrix,TestingValues )



#import matplotlib.pyplot as plt
plt.title('Predicted values compared to actual values - test set.' )
plt.xlabel('ground truth')
plt.ylabel("prediction")
plt.scatter(TestingValues, Predicted_Values)
plt.show()
