from NeuralNetInNumpy import *

#DesignMatrix,TrainingValues= np.matrix([-1, 10,-10,-15, -20, 50,  20, 17]), np.matrix([0, 1, 0, 0, 0,  1, 1, 1])
DesignMatrix,TrainingValues= GenerateTrainingData(1,11)
X=np.transpose(DesignMatrix)  # make column vector inputs
Y=np.transpose(TrainingValues)
print(X,Y)
network = [dense_layer(1,6), relu_layer(), dense_layer(6,1)]
# relu doesn't have nodes to define. Data is passed in relu_layer.forward(data) call, when the network is evaluated. 
print(X,Y)
epochs = 600
learning_rate = 0.03
# define the error function
error_function=mse  
error_grad=mse_gradient
RunNetwork(epochs, X, Y, learning_rate, error_function, error_grad, network)

DesignMatrix,TestingValues= GenerateTrainingData(30,41)
PredictWithNetwork(DesignMatrix, network)
print("actual values:", DesignMatrix,TestingValues )