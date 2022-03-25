
from NeuralNetInNumpy import *


#DesignMatrix,TrainingValues= np.matrix([-1, 10,-10,-15, -20, 50,  20, 17]), np.matrix([0, 1, 0, 0, 0,  1, 1, 1])
DesignMatrix,TrainingValues= GenerateTrainingData(1,11)
X=np.transpose(DesignMatrix)  # make column vector inputs
Y=np.transpose(TrainingValues)
print(X,Y)
network = [dense_layer(1,6), sigmoid_layer(), dense_layer(6,1)]
# sigmoid doesn't have nodes to define. Data is passed in sigmoid_layer.forward(data) call, when the network is evaluated. 
#scale_factor = np.std(X)
#X = X/scale_factor
print(X,Y)
epochs = 2000
learning_rate = 0.000000005
# define the error function
error_function=mse  
error_grad=mse_gradient
RunNetwork(epochs, X, Y, learning_rate, error_function, error_grad, network)

#DesignMatrix,TestingValues= np.matrix([-10001,-510, 1, 10000, -100000, -2000000,  2, 7]), np.matrix([0, 0, 1, 1, 0, 0,  1, 1])
DesignMatrix,TestingValues= GenerateTrainingData(30,41)
PredictWithNetwork(DesignMatrix, network)
print("actual values:", DesignMatrix,TestingValues )