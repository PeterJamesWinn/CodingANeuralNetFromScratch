from NeuralNetInNumpy import *
import matplotlib.pyplot as plt
# relu not very good at this sort of classification, but results show 
# separation. 

# Data for training.
DesignMatrix,TrainingValues = np.array([[50, -1, 10,-10,-15, -20,   20, 17]]), np.array([[1, 0, 1, 0, 0, 0,  1, 1]]) 

# Reformatting data
X=np.transpose(DesignMatrix)  # make column vector inputs - need this format for stochastic gradient descent. 
Y=np.transpose(TrainingValues)

print("X, Y: \n", X, "\n",Y)
normalising_factor = np.std(X) # rescale data for classification problem. Not for regression. 
print("normalising_factor", normalising_factor, "\n")
X = X/normalising_factor
print("normalised X, Y: \n", X, "\n", Y, "\n")

#Network topology
network = [dense_layer(1,6), relu_layer(), dense_layer(6,6), relu_layer() , \
           dense_layer(6,6),relu_layer() , dense_layer(6,1), sigmoid_layer()]
#network = [dense_layer(1,6), relu_layer(), dense_layer(6,1), sigmoid_layer()]
# relu doesn't have nodes to define. Data is passed in 
# relu_layer.forward(data) call, when the network is evaluated. 

# Training parameters
epochs = 200
learning_rate = 0.000003
# define the error function
error_function = binary_cross_entropy  
error_grad = binary_cross_entropy_gradient

# Train network.
print("\n Training Network \n \n ")
RunNetwork(epochs, X, Y, learning_rate, error_function, error_grad, network)

# Test network.
DesignMatrix,TestingValues = np.array([[50, -1, 10,-10,-15, -20,   20, 17]]), np.array([[ 1, 0, 1, 0, 0, 0,  1, 1]])
X = np.transpose(DesignMatrix)  # make column vector inputs
X = X/normalising_factor
#DesignMatrix,TestingValues= GenerateTrainingData(30,41) # regression problem testing. 

Predicted_Values = PredictWithNetwork(X, network)
print("predictions:", DesignMatrix, Predicted_Values)
print("actual values:", DesignMatrix,TestingValues )
print(np.asarray(TestingValues).flatten(), Predicted_Values)
#np.asarray(DesignMatrixString1).reshape((1,(max-min)))

plt.title('Predicted values compared to actual values - training set.' )
plt.xlabel('ground truth')
plt.ylabel("prediction")
plt.scatter(np.asarray(TestingValues).flatten(), Predicted_Values)
plt.show()

DesignMatrix,TestingValues = \
    np.array([[-100,-510, 1, 100, -1000, -2,  29, 67]]), \
        np.array([[0, 0, 1, 1, 0, 0,  1, 1]])
X = np.transpose(DesignMatrix)  # make column vector inputs
Y = np.transpose(TestingValues)
X = X/normalising_factor
#DesignMatrix,TestingValues= GenerateTrainingData(30,41)
Predicted_Values=PredictWithNetwork(X, network)
print("predictions:", DesignMatrix, Predicted_Values)
print("actual values:", DesignMatrix,TestingValues )
plt.title('Predicted values compared to actual values - test set.' )
plt.xlabel('ground truth')
plt.ylabel("prediction")
plt.scatter(TestingValues, Predicted_Values)
plt.show()

