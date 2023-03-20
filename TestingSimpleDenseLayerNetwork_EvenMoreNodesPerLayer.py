
from NeuralNetInNumpy import *


#DesignMatrix,TrainingValues= np.matrix([-1, 10,-10,-100, -20, 100,  20, 107]), np.matrix([0, 1, 0, 0, 0,  1, 1, 1])
DesignMatrix,TrainingValues= GenerateTrainingData(1,11)
#network = [dense_layer(), sigmoid_layer(), dense_layer()]
X=np.transpose(DesignMatrix)  # make column vector inputs
Y=np.transpose(TrainingValues)
print(X,Y)
network = [dense_layer(1,6),  dense_layer(6,1)]
#network = [dense_layer(1,3), sigmoid_layer(), dense_layer(3,1)] # sigmoid doesn't have nodes to define. Data is passed in sigmoid_layer.forward(data) call, when the network is evaluated. 
#for x in X:
#  x = x/np.max(abs(X))  # normalise data
#  print("normalised x: ", x)
print(X,Y)
epochs = 500
learning_rate = 0.001
# define the error function
error_function=mse  
error_grad=mse_gradient
RunNetwork(epochs, X, Y, learning_rate, error_function, error_grad, network)
print("finished training")

DesignMatrix,TestingValues = GenerateTrainingData(30,41)
X=np.transpose(DesignMatrix)  # make column vector inputs
Y=np.transpose(TrainingValues)
Predicted_Values = PredictWithNetwork(X, network)
print("predictions:", DesignMatrix, Predicted_Values)
print("actual values:", DesignMatrix,TestingValues )