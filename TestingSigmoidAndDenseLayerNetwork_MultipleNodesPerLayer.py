from NeuralNetInNumpy import *
# with sigmoid the learning of a simple linear function is much worse 
# than with only one linear layer, or with linear plus relu. Presumably
# because of squashing data into values between 0 and 1. 


DesignMatrix, TrainingValues = GenerateTrainingData(1,11)
X = np.transpose(DesignMatrix)  # make column vector inputs
Y = np.transpose(TrainingValues)
print(X,Y)
network = [dense_layer(1,6), sigmoid_layer(), \
                                dense_layer(6,6), dense_layer(6,1)]
# sigmoid doesn't have nodes to define. Data is passed in 
# sigmoid_layer.forward(data) call, when the network is evaluated. 
print(X,Y)
epochs = 300
learning_rate = 0.0001
# define the error function
error_function = mse  
error_grad = mse_gradient
batch_size = 2
RunNetwork_BatchOptimisation(epochs, X, Y, learning_rate, \
                             error_function, error_grad, network, batch_size)

DesignMatrix, TestingValues = GenerateTrainingData(30,41)
X = np.transpose(DesignMatrix)  # make column vector inputs
Y = np.transpose(TrainingValues)
Predicted_Values=PredictWithNetwork(X, network)
print("predictions:", DesignMatrix, Predicted_Values)
print("actual values:", DesignMatrix,TestingValues )