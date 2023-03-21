from NeuralNetInNumpy import *

# testing modules.
def test_mse():
  print("test the mean squared error loss function (mse())\n")
  print("mse(1, 9, 1) = ", mse(1, 9, 1))
  assert mse(1, 9, 1) == 64, "MSE check failed on check 1"
  print("MSE passed basic check 1") 
  assert mse(9, 9, 1) == 0, "MSE check failed on check 2"
  print("MSE passed basic check 2"), 
  assert mse (9, 1, 1) == 64, "MSE check failed on check 3"
  print("MSE passed basic check 3. \n")
  print("Two further MSE checks for manual inspection:\n") 
  #for y, y_hat in [(1,1), (10,1)]:
  #print("y_hat: {}\n y: {} \n mse: {}\n".format(y_hat, y, mse(y, y_hat, 1)))
  return

def test_mse_gradient():  
  def finite_diff_grad(y, y_hat):
    if y != 0:
        increment = y * 0.00001
    else: increment = 0.00000001
    rounding_tolerance = abs(increment)
    diff = mse(y, (y_hat + increment), 1) - mse(y, (y_hat - increment), 1)
    # we're ignoring the factor of 2 in the mse_gradient calculation that
    # comes from differentiating the square, so include an extra 0.5 
    # factor on the finite difference calculation 
    # to be compatible with that.
    return 0.5 * (diff/(2 * increment)), rounding_tolerance

  print("test the calculation of the gradient of",\
        "the mean squared error loss function (mse_gradient())\n")
  print("mse_gradient, finite_difference_gradient: ", \
        mse_gradient(9, 1, 1), finite_diff_grad(9, 1)[0])
  assert mse_gradient(9, 1, 1) - finite_diff_grad(9,1)[0] \
    < finite_diff_grad(9,1)[1] , "mse_gradient check failed on check 1"
  print("mse_gradient passed basic check 1")
  print("check 2")
  print("mse_gradient, finite_difference_gradient: ", \
        mse_gradient(1, 9, 1), finite_diff_grad(1, 9)[0])
  assert mse_gradient(1, 9, 1) - finite_diff_grad(1, 9)[0] \
    < finite_diff_grad(9,1)[1] , "mse_gradient check failed on check 2"
  print("mse_gradient passed basic check 2.\n")
  return

def test_sigmoid():
  assert sigmoid(0) == 0.5, "sigmoid(0) fails test"
  print("sigmoid(0) = 0.5 correctly calculated")
  assert sigmoid(1000) == 1, "sigmoid(1000) fails test"
  print("sigmoid(1000) = 1 correctly calculated")
  assert sigmoid(-1000) == 0, "sigmoid(-1000) fails test"
  print("sigmoid(-1000) = 0 correctly calculated")
  assert abs(sigmoid(1) - 0.731) < 0.001, "sigmoid(1) fails test"
  print("sigmoid(1) = 0.731 correctly calculated")
  assert abs(sigmoid(-1) - 0.269) < 0.001, "sigmoid(-1) fails test"
  print("sigmoid(-1) = 0.269 correctly calculated")

def test_sigmoid_gradient():
  def finite_diff_grad(y):
    if y != 0:
        increment = y * 0.00001
    else: increment = 0.00000001
    rounding_tolerance = abs(increment)
    diff = sigmoid(y+ increment) - sigmoid(y - increment)
    return (diff/(2 * increment)), rounding_tolerance

  print("test the calculation of the gradient of",\
        "the sigmoid function (sigmoid_gradient())\n")
  print("sigmoid_gradient, finite_difference_gradient: ", \
        sigmoid_gradient(9), finite_diff_grad(9)[0])
  assert sigmoid_gradient(9) - finite_diff_grad(9)[0] \
    < finite_diff_grad(9)[1] , "sigmoid_gradient(9) check failed on check 1"
  print("sigmoid_gradient(9) passed basic check 1")
  print("check 2")
  print("sigmoid_gradient, finite_difference_gradient: ", \
        sigmoid_gradient(0), finite_diff_grad(0)[0])
  assert sigmoid_gradient(0) - finite_diff_grad(0)[0] \
    < finite_diff_grad(0)[1] , "sigmoid_gradient(9) check failed on check 2"
  print("sigmoid_gradient(9) passed basic check 2")
  print("check 2")

def test_sigmoid_gradient_sig_arg():
  '''
  Takes a numberical value calculates its sigmoid and uses that to test
  sigmoid_gradient_sig_arg(), which takes the output of a sigmoid 
  function as input.
  '''
  def finite_diff_grad(y):
    if y != 0:
        increment = y * 0.001
    else: increment = 0.00000001
    rounding_tolerance = abs(increment)
    diff = sigmoid(y+ increment) - sigmoid(y - increment)
    return (diff/(2 * increment)), rounding_tolerance
  
  def perform_check(x):
    print("test the calculation of the gradient of",\
        " the sigmoid function (sigmoid_gradient_sig_arg())", \
          ", taking a sigmoid as an argument - to work with"\
          " the sigmoid layer class. \n")
    print("sigmoid_gradient_sig_arg(", x, "), finite_difference_gradient: ", \
          sigmoid_gradient_sig_arg(sigmoid(x)), finite_diff_grad(x)[0])
    assert sigmoid_gradient(x) - finite_diff_grad(x)[0] \
                                < finite_diff_grad(x)[1] ,\
            "sigmoid_gradient_sig_arg(sigmoid()) check failed"
    print("sigmoid_gradient_sig_arg(sigmoid(", x, ")) passed basic check")

  perform_check(-10)
  perform_check(-2)
  perform_check(9)
  perform_check(0)

def test_sigmoid_layer_backprop():
  print("\n testing sigmoid backprop")
  test = sigmoid_layer()
  test.sigmoid = 2
  local_gradient = sigmoid_gradient_sig_arg(test.sigmoid)
  upstream_gradient = np.array([1,2,3])
  learning_rate = 0.1 # not needed a sigmoid layer but for the generality 
                      # of layer implementation
  downstream_gradient = test.backward_pass(upstream_gradient, learning_rate)
  print("local gradient: {} \n value of sigmoid function: \
         {} \n".format(local_gradient, test.sigmoid)) 
  print("upstream gradent: {} \n downstream grad.: \
        {}\n".format(upstream_gradient,downstream_gradient ))

  print("\n testing sigmoid backprop")
  test = sigmoid_layer()
  test.sigmoid = sigmoid(np.array([[2, 4,5]])) #vector in; vector out.
  print("test.sigmoid", test.sigmoid)
  local_gradient = sigmoid_gradient_sig_arg(test.sigmoid)
  upstream_gradient = np.array([1,2,3])
  learning_rate = 0.1 # not needed a sigmoid layer but for the generality 
                      # of layer implementation
  downstream_gradient = test.backward_pass(upstream_gradient, learning_rate)
  print("local gradient: {} \n value of sigmoid function:\
         {} \n".format(local_gradient, test.sigmoid)) 
  print("upstream gradent: {} \n downstream grad: \
        {}\n".format(upstream_gradient,downstream_gradient ))


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
test_sigmoid_gradient()
test_sigmoid_gradient_sig_arg()


test_mse()
test_mse_gradient()
test_sigmoid_layer_backprop()




