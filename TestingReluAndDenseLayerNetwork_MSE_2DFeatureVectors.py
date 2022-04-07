

from NeuralNetInNumpy import *
import matplotlib.pyplot as plt

# Data for training.
#DesignMatrix,TrainingValues= np.matrix([50, -1, 10,-10,-15, -20,   20, 17]), np.matrix([ 1, 0, 1, 0, 0, 0,  1, 1])  # if independent variable < 0 then out is 0, if > 0 then 1
#DesignMatrix,TrainingValues= GenerateTrainingData(1,4) # regression problem training. 
DesignMatrix1, DesignMatrix2, TrainingValues= GenerateTrainingData_2DFeature(1,10)
X = np.concatenate((np.transpose(DesignMatrix1), np.transpose(DesignMatrix2)), axis=1)
print("X", X)

# Reformatting data
#X=np.transpose(X)  # make column vector inputs - need this format for stochastic gradient descent. 
Y=np.transpose(TrainingValues)
#X = DesignMatrix
#Y = TrainingValues
print("X, Y, data generation: ", X, "\n",Y)

# Input Normalisation for classification problems
#factor = np.std(X) # rescale data for classification problem. Not for regression. 
#print("factor", factor)
#X = X/factor
#print(X,Y)

#Network topology
network = [dense_layer(2,6), relu_layer(), dense_layer(6,6), relu_layer() , dense_layer(6,6),relu_layer() , dense_layer(6,1)]
#network = [dense_layer(2,6), relu_layer(),  dense_layer(6,1)]
#network = [dense_layer(1,6), relu_layer(), dense_layer(6,1), relu_layer()]
# relu doesn't have nodes to define. Data is passed in relu_layer.forward(data) call, when the network is evaluated. 
#X -= np.mean(X, axis=0)

# Training parameters
epochs = 30
learning_rate = 0.0003
# define the error function
error_function=mse
error_grad=mse_gradient

#error_function=binary_cross_entropy  
#error_grad=binary_cross_entropy_gradient

# Train network.
#print(Y, len(Y))
RunNetwork(epochs, X, Y, learning_rate, error_function, error_grad, network)
 


# Test network.
# using training data
X_Test = np.concatenate((np.transpose(DesignMatrix1), np.transpose(DesignMatrix2)), axis=1)
print("X", X)
Y=np.transpose(TrainingValues)
Predicted_Values_TrainingData=PredictWithNetwork(X, network) # Using training data.
print("predictions on training data:", X, Predicted_Values_TrainingData)
print("actual values:", X, TrainingValues )
#plt.title('Predicted values compared to actual values - training set.' )
#plt.xlabel('ground truth')
#plt.ylabel("prediction")
#plt.scatter(TestingValues, Predicted_Values)
#plt.show()


DesignMatrix1, DesignMatrix2, TestingValues = GenerateTrainingData_2DFeature(40,50)
X_test = np.concatenate((np.transpose(DesignMatrix1), np.transpose(DesignMatrix2)), axis=1)
print("X", X)
Predicted_Values=PredictWithNetwork(X_test, network)
print("predictions on testing data :", X_test, Predicted_Values)
print("actual values:", X_test, TestingValues)
print("predictions on training data:", X, Predicted_Values_TrainingData)
print("actual values:", X, TrainingValues )



