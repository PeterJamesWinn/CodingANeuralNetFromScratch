from NeuralNetInNumpy import *
import matplotlib.pyplot as plt

# Data for training.
# Regression problem.
# Below, GenerateTrainingData(1,10) creates an array of Xs from 1 to 3,
# inclusive, and uses the function called in GenerateTrainingData 
# to generate the Y values - e.g.  Y = 5.0 + DesignMatrix * 3.0.
# Future iterations of the code should pass the function to
# GenerateTrainingData as part of the call. 
DesignMatrix, TrainingValues = GenerateTrainingData(1,10)  

# Reformatting data
# make column vector inputs -  format for stochastic gradient descent. 
X = np.transpose(DesignMatrix)  
Y = np.transpose(TrainingValues)
print("X, Y training data:", X, "\n",Y)

#Network topology
#network = [dense_layer(1,6, "relu"), relu_layer(), dense_layer(6,6, "relu"), \
#          relu_layer() , dense_layer(6,6, "relu"), relu_layer()\
#             , dense_layer(6,6, "relu"),relu_layer(), dense_layer(6,1)]
network = [dense_layer(1,6, "relu"), relu_layer(), dense_layer(6,6, "relu"), \
          relu_layer() , dense_layer(6,6, "relu"), relu_layer()\
             , dense_layer(6,1)]
#network = [dense_layer(1,6, "relu"), relu_layer(),  dense_layer(6,1)]
#network = [dense_layer(1,6, "relu"), relu_layer(), dense_layer(6,6, "relu"),
#                                 relu_layer() , dense_layer(6,1)]
# relu doesn't have nodes to define. Data is passed 
# in relu_layer.forward(data) call, when the network is evaluated. 

# Training parameters
epochs = 4000
learning_rate = 0.0002
# define the error function
error_function = mse
error_grad = mse_gradient

#error_function=binary_cross_entropy  
#error_grad=binary_cross_entropy_gradient

# Train network.
RunNetwork(epochs, X, Y, learning_rate, error_function, error_grad, network)
 
# Test network.
DesignMatrix, TestingValues = GenerateTrainingData(30,41) #regression test.
X = np.transpose(DesignMatrix)  # make column vector inputs
 
Predicted_Values=PredictWithNetwork(X, network)
print("predictions:", DesignMatrix, Predicted_Values)
print("actual values:", DesignMatrix,TestingValues )
print("MSE: ", np.sum(mse(TestingValues, Predicted_Values, len(TestingValues))), "\n")
plt.title('Predicted values compared to actual values - training set.' )
plt.xlabel('ground truth')
plt.ylabel("prediction")
plt.scatter(TestingValues, Predicted_Values)
plt.show()



