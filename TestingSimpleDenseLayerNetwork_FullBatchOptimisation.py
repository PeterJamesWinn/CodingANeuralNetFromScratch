


from NeuralNetInNumpy import *

# test network. The weight matrix and the bias recover the equation of the line used to generate the data,
# albeit the bias being 4.8 instead of 5, the gradient 3.01 instead of 3. Nonetheless the test shows the network
# works OK. Need an alternative optimisation process compared to steepest gradient descent. 
network = [ dense_layer(1,1)]
DesignMatrix,TrainingValues= GenerateTrainingData(1,11)
#network = [dense_layer(), sigmoid_layer(), dense_layer()]
#X=np.transpose(DesignMatrix)  # make column vector inputs
#Y=np.transpose(TrainingValues)
X = DesignMatrix
Y = TrainingValues
print(X,Y)
epochs = 1
learning_rate = 0.1
# define the error function
error_function=mse  
error_grad=mse_gradient
RunNetwork_BatchOptimisation(epochs, X, Y, learning_rate, error_function, error_grad, network)

DesignMatrix,TestingValues= GenerateTrainingData(30,41)
Predicted_Values=PredictWithNetwork(DesignMatrix, network)
print("predictions:", DesignMatrix, Predicted_Values)
print("actual values:", DesignMatrix,TestingValues )
