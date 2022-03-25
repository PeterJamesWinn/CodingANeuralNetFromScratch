

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
  learning_rate = 0.1 # not actually needed for a sigmoid layer but needed for the generality of layer implementation
  downstream_gradient = test.backward_pass(upstream_gradient, learning_rate)
  print("local gradient: {} \n value of sigmoid function: {} \n".format(local_gradient, test.sigmoid)) 
  print("upstream gradent: {} \n downstream grad. :{}\n".format(upstream_gradient,downstream_gradient ))

  print("\n testing sigmoid backprop")
  test = sigmoid_layer()
  test.sigmoid = sigmoid(np.array([[2, 4,5]]))  # vector into sigmoid gives vector out.
  print("test.sigmoid", test.sigmoid)
  local_gradient = (1 - test.sigmoid)/test.sigmoid
  upstream_gradient = np.array([1,2,3])
  learning_rate = 0.1 # not actually needed for a sigmoid layer but needed for the generality of layer implementation
  downstream_gradient = test.backward_pass(upstream_gradient, learning_rate)
  print("local gradient: {} \n value of sigmoid function: {} \n".format(local_gradient, test.sigmoid)) 
  print("upstream gradent: {} \n downstream grad. :{}\n".format(upstream_gradient,downstream_gradient ))


# Test layers. 
'''
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
'''

#print(dense_layer.backward_pass.__doc__)   
test_sigmoid()
test_sigmoid_layer_backprop()

'''
test_mse()
test_mse_gradient()
'''




