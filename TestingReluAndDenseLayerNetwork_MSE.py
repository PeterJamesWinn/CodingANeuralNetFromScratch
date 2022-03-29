

from NeuralNetInNumpy import *
import matplotlib.pyplot as plt

# Data for training.
#DesignMatrix,TrainingValues= np.matrix([50, -1, 10,-10,-15, -20,   20, 17]), np.matrix([ 1, 0, 1, 0, 0, 0,  1, 1])  # if independent variable < 0 then out is 0, if > 0 then 1
DesignMatrix,TrainingValues= GenerateTrainingData(1,11) # regression problem training. 

# Reformatting data
X=np.transpose(DesignMatrix)  # make column vector inputs - need this format for stochastic gradient descent. 
Y=np.transpose(TrainingValues)
#X = DesignMatrix
#Y = TrainingValues
print(X, "\n",Y)

# Input Normalisation for classification problems
#factor = np.std(X) # rescale data for classification problem. Not for regression. 
#print("factor", factor)
#X = X/factor
#print(X,Y)

#Network topology
network = [dense_layer(1,6), relu_layer(), dense_layer(6,6), relu_layer() , dense_layer(6,6),relu_layer() , dense_layer(6,1)]
#network = [dense_layer(1,6), relu_layer(), dense_layer(6,1), relu_layer()]
# relu doesn't have nodes to define. Data is passed in relu_layer.forward(data) call, when the network is evaluated. 
#X -= np.mean(X, axis=0)

# Training parameters
epochs = 2500
learning_rate = 0.0003
# define the error function
error_function=mse
error_grad=mse_gradient

#error_function=binary_cross_entropy  
#error_grad=binary_cross_entropy_gradient

# Train network.
RunNetwork(epochs, X, Y, learning_rate, error_function, error_grad, network)



# Test network.
DesignMatrix,TestingValues= GenerateTrainingData(30,41) # regression problem testing.
#DesignMatrix,TestingValues= np.array([50, -1, 10,-10,-15, -20,   20, 17]), np.array([ 1, 0, 1, 0, 0, 0,  1, 1])
X=np.transpose(DesignMatrix)  # make column vector inputs
#Y=np.transpose(TestingValues)
#X = DesignMatrix
#Y = TestingValues
#X= X/factor
 
Predicted_Values=PredictWithNetwork(X, network)
print("predictions:", DesignMatrix, Predicted_Values)
print("actual values:", DesignMatrix,TestingValues )
plt.title('Predicted values compared to actual values - training set.' )
plt.xlabel('ground truth')
plt.ylabel("prediction")
plt.scatter(TestingValues, Predicted_Values)
plt.show()


