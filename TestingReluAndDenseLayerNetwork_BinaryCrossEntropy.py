

from NeuralNetInNumpy import *

DesignMatrix,TrainingValues= np.matrix([50, -1, 10,-10,-15, -20,   20, 17]), np.matrix([ 1, 0, 1, 0, 0, 0,  1, 1])
#DesignMatrix,TrainingValues= GenerateTrainingData(1,11)
X=np.transpose(DesignMatrix)  # make column vector inputs
Y=np.transpose(TrainingValues)
print(X,Y)
network = [dense_layer(1,6), relu_layer(), dense_layer(6,6), relu_layer() , dense_layer(6,6),relu_layer() , dense_layer(6,1)]
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
RunNetwork(epochs, X, Y, learning_rate, error_function, error_grad, network)
#DesignMatrix,TestingValues= np.matrix([-100,-510, 1, 100, -1000, -2,  29, 67]), np.matrix([0, 0, 1, 1, 0, 0,  1, 1])
DesignMatrix,TestingValues= np.matrix([50, -1, 10,-10,-15, -20,   20, 17]), np.matrix([ 1, 0, 1, 0, 0, 0,  1, 1])
X= X/factor
#DesignMatrix,TestingValues= GenerateTrainingData(30,41)
PredictWithNetwork(DesignMatrix, network)
print("actual values:", DesignMatrix,TestingValues )

DesignMatrix,TestingValues= np.matrix([-100,-510, 1, 100, -1000, -2,  29, 67]), np.matrix([0, 0, 1, 1, 0, 0,  1, 1])
X= X/factor
#DesignMatrix,TestingValues= GenerateTrainingData(30,41)
PredictWithNetwork(DesignMatrix, network)
print("actual values:", DesignMatrix,TestingValues )