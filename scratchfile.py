
import matplotlib.pyplot as plt
import numpy as np
plt.xlabel('ground truth')
plt.ylabel("prediction")
plt.scatter([50, -1, 10,-10,-15, -20,   20, 17],[ 1, 0, 1, 0, 0, 0,  1, 1])
plt.show()

'''
# loss functions aka error functions:
def mse(y, y_hat): 
  mse: mean squared error loss: 0.5(y_hat - y)squared. y_hat is the estimate from the network, y is the ground truth
  return 0.5*np.square(y_hat - y)

def mse_gradient(y, y_hat):
  gradient of mse: mean squared error loss: (y_hat - y). y_hat is the estimate from the network, y is the ground truth
  return y_hat - y

DesignMatrix,TestingValues= np.matrix([50, -1, 10,-10,-15, -20,   20, 17]), np.matrix([ 1, 0, 1, 0, 0, 0,  1, 1])
DesignMatrix,TestingValues= np.matrix([1, 1, 1, 1, 1, 1, 1, 2]), np.matrix([ 0, 0, 0, 0, 0, 0,  0, 0])
l = mse(TestingValues, DesignMatrix)
print("loss1 : ", l)

X=np.transpose(DesignMatrix)  # make column vector inputs
Y=np.transpose(TestingValues)
print("Design Matrix, Training Values", DesignMatrix, TestingValues)
l = mse(X, Y)
print("loss 2: ", l)

X=np.transpose(DesignMatrix)  # make column vector inputs
Y=np.transpose(TestingValues)
for x, y in zip(X,Y):
    print("Design Matrix, Training Values", DesignMatrix, TestingValues)
    l = mse(x, y)
    grad = mse_gradient
    print("loss 3: ", l)


x= np.asarray([[ 50,  -1,  10, -10, -15, -20, 20,  17]])
print(x)
for y in x:
    print(y)
for y in x[0:]:
    print(y)
for y in x[0:,]:
    print(y)
'''