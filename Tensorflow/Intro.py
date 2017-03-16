import numpy as np
import tensorflow as tf

# specify the type of placeholderss
A = tf.placeholder(tf.float32, shape=(5, 5), name='A')
v = tf.placeholder(tf.float32)

# matrix multiplication
w = tf.matmul(A, v)

with tf.Session() as sess:
	output = sess.run(w, feed_dict={A: np.random.randn(5, 5), v: np.random.randn(5, 1)})
	print(output, type(output))

# TensorFlow variables are like Theano shared variables.
# But Theano variables are like TensorFlow placeholders.

# A tf variable can be initialized with a numpy array or a tf array
# or more correctly, anything that can be turned into a tf tensor
shape = (2, 2)
x = tf.Variable(tf.random_normal(shape))
# x = tf.Variable(np.random.randn(2, 2))
t = tf.Variable(0)  # a scalar

# you need to "initialize" the variables first
init = tf.global_variables_initializer()

with tf.Session() as session:
	out = session.run(init)  # and then "run" the init operation
	print(out)
	print(x.eval())
	print(t.eval())

# let's now try to find the minimum of a simple cost function like we did in Theano
u = tf.Variable(20.0)
cost = u * u + u + 1.0

# One difference between Theano and TensorFlow is that you don't write the updates
# yourself in TensorFlow. You choose an optimizer that implements the algorithm you want.
# 0.3 is the learning rate. Documentation lists the params.
train_op = tf.train.GradientDescentOptimizer(0.3).minimize(cost)

# let's run a session again
init = tf.global_variables_initializer()
with tf.Session() as session:
	session.run(init)

	# Strangely, while the weight update is automated, the loop itself is not.
	# So we'll just call train_op until convergence.
	# This is useful for us anyway since we want to track the cost function.
	for i in range(12):
		session.run(train_op)
		print("i = %d, cost = %.3f, u = %.3f" % (i, cost.eval(), u.eval()))

	print(u.eval())
