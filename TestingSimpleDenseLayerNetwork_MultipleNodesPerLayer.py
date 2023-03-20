from NeuralNetInNumpy import *

DesignMatrix, TrainingValues = GenerateTrainingData(1,11)
X = np.transpose(DesignMatrix)  # make column vector inputs
Y = np.transpose(TrainingValues)
print(X,Y)
network = [dense_layer(1,3),  dense_layer(3,1)]
print(X,Y)
epochs = 600
learning_rate = 0.001
# define the error function
error_function = mse  
error_grad = mse_gradient
RunNetwork(epochs, X, Y, learning_rate, error_function, error_grad, network)
print("finished training")

DesignMatrix, TestingValues = GenerateTrainingData(30,41)
X=np.transpose(DesignMatrix)  # make column vector inputs
Y=np.transpose(TrainingValues)
PredictWithNetwork(X, network)
print("actual values:", DesignMatrix,TestingValues )

