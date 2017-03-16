import theano.tensor as T

# different type of theano variables
c = T.scalar('c')
v = T.vector('v')
A = T.matrix('A')

# define some algebra for theano variables
w = A.dot(v)

import theano

# define theano function
matrix_times_vector = theano.function(inputs=[A, v], outputs=w)

import numpy as np

A_val = np.array([[1, 2], [3, 4]])
v_val = np.array([5, 6])

w_val = matrix_times_vector(A_val, v_val)
print(w_val)

# create a shared (updatable) variable
x = theano.shared(20.0, 'x')

# create a simple cost with an obvious minimum
cost = x * x + x + 1

# update expression for x
x_update = x - 0.3 * T.grad(cost, x)

# define the train function
train = theano.function(inputs=[], outputs=[cost], updates=[(x, x_update)])

# epoch training
for i in range(25):
	cost_val = train()
	print(cost_val)

# print the optimal value of x
print(x.get_value())
