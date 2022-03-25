

from NeuralNetInNumpy import *

# testing modules.
def test_mse():
  print("test the mean squared error loss function (mse())\n")
  for y, y_hat in [(1,1), (10,1)]:
    print("y_hat: {}\n y: {} \n mse: {}\n".format(y_hat, y, mse(y, y_hat)))
  return

def test_mse_gradient():
  print("test the calculation of the gradient of the mean squared error loss function (mse_gradient())\n")
  for y, y_hat in [(1,1), (10,1)]:
    print("y_hat: {}\n y: {} \n mse_gradient: {}\n".format(y_hat, y, mse_gradient(y, y_hat)))
  return


def test_sigmoid():
  y = sigmoid(0)
  z = sigmoid(1000)
  omega = sigmoid(-1000)
  a = sigmoid(1)
  b = sigmoid(-1)
  print("test sigmoid inputs: 0, 1000, -1000, 1, -1: ouputs:", y, z, omega, a, b)


def test_sigmoid_layer_backprop():
  print("\n testing sigmoid backprop")
  test = sigmoid_layer()
  test.sigmoid = 2
  local_gradient = (1 - test.sigmoid)/test.sigmoid
  upstream_gradient = np.array([1,2,3])
  downstream_gradient = test.backward_pass(upstream_gradient)
  print("local gradient: {} \n value of sigmoid function: {} \n".format(local_gradient, test.sigmoid)) 
  print("upstream gradent: {} \n downstream grad. :{}\n".format(upstream_gradient,downstream_gradient ))

'''
# Test layers. 
l0 = dense_layer(2,1)
print("bias: ", l0.bias)
print("weights: ", l0.weights)
forward = l0.forward_pass([[1],[1]])
print("feature_vector: ", l0.feature_vector)
print(forward)
gradient= l0.backward_pass(1, 0.1)
print("gradient:", gradient)

l0 = dense_layer(2,1)
print("bias: ", l0.bias)
print("weights: ", l0.weights)
forward = l0.forward_pass([[1],[2]])
print("feature_vector: ", l0.feature_vector)
print(forward)
gradient= l0.backward_pass(2, 0.1)
print("gradient:", gradient)

print(dense_layer.backward_pass.__doc__)   
test_sigmoid()
test_sigmoid_layer_backprop()
test_mse()
test_mse_gradient()
'''

'''
# test network. The weight matrix and the bias recover the equation of the line used to generate the data,
# albeit the bias being 4.8 instead of 5, the gradient 3.01 instead of 3. Nonetheless the test shows the network
# works OK. Need an alternative optimisation process compared to steepest gradient descent. 
network = [ dense_layer(1,1)]
DesignMatrix,TrainingValues= GenerateTrainingData(1,11)
#network = [dense_layer(), sigmoid_layer(), dense_layer()]
X=np.transpose(DesignMatrix)  # make column vector inputs
Y=np.transpose(TrainingValues)
print(X,Y)
epochs = 100
learning_rate = 0.1
# define the error function
error_function=mse  
error_grad=mse_gradient
RunNetwork(epochs, X, Y, learning_rate, error_function, error_grad, network)

DesignMatrix,TestingValues= GenerateTrainingData(30,41)
PredictWithNetwork(DesignMatrix, network)
print("actual values:", DesignMatrix,TestingValues )
'''


#DesignMatrix,TrainingValues= np.matrix([-1, 10,-10,-100, -20, 100,  20, 107]), np.matrix([0, 1, 0, 0, 0,  1, 1, 1])
DesignMatrix,TrainingValues= GenerateTrainingData(1,11)
#network = [dense_layer(), sigmoid_layer(), dense_layer()]
X=np.transpose(DesignMatrix)  # make column vector inputs
Y=np.transpose(TrainingValues)
print(X,Y)
network = [dense_layer(1,3),  dense_layer(3,1)]
#network = [dense_layer(1,3), sigmoid_layer(), dense_layer(3,1)] # sigmoid doesn't have nodes to define. Data is passed in sigmoid_layer.forward(data) call, when the network is evaluated. 
X=np.transpose(DesignMatrix)  # make column vector inputs
#for x in X:
#  x = x/np.max(abs(X))  # normalise data
#  print("normalised x: ", x)
Y=np.transpose(TrainingValues)
print(X,Y)
epochs = 200
learning_rate = 0.01
# define the error function
error_function=mse  
error_grad=mse_gradient
RunNetwork(epochs, X, Y, learning_rate, error_function, error_grad, network)

#DesignMatrix,TestingValues= np.matrix([-10001,-510, 1, 10000, -100000, -2000000,  2, 7]), np.matrix([0, 0, 1, 1, 0, 0,  1, 1])
#for x in DesignMatrix:
#  x = x/np.max(abs(DesignMatrix))  # normalise data
#  print("normalised x: ", x)
DesignMatrix,TestingValues= GenerateTrainingData(30,41)
PredictWithNetwork(DesignMatrix, network)
print("actual values:", DesignMatrix,TestingValues )
