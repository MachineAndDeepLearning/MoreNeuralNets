import numpy as np
import theano
import theano.tensor as T
import matplotlib.pyplot as plt

from Util import MNIST, y2indicator


def error_rate(p, t):
    return np.mean(p != t)

# define relu in case the version of theano doesnt have it
def relu(a):
	return a * (a > 0)


def main():
	X, Y = MNIST().get_normalized_data(up_one_level=True)

	max_iter = 20
	print_period = 10
	lr = 0.00004
	reg = 0.01

	# split the data and initialize the weights and biases
	Xtrain, Ytrain = X[:-1000, ], Y[:-1000]
	Xtest, Ytest = X[-1000:], Y[-1000:]
	Ytrain_ind, Ytest_ind = y2indicator(Ytrain), y2indicator(Ytest)

	N, D = Xtrain.shape
	batch_sz = 500
	n_batches = N // batch_sz

	M = 300
	K = 10
	W1_init = np.random.randn(D, M) / 28
	b1_init = np.zeros(M)
	W2_init = np.random.randn(M, K) / np.sqrt(M)
	b2_init = np.zeros(K)

	# create theano variables
	thX = T.matrix('X')
	thT = T.matrix('T')

	W1 = theano.shared(W1_init, 'W1')
	b1 = theano.shared(b1_init, 'b1')
	W2 = theano.shared(W2_init, 'W2')
	b2 = theano.shared(b2_init, 'b2')

	thZ = T.nnet.relu(thX.dot(W1) + b1)
	thY = T.nnet.softmax(thZ.dot(W2) + b2)

	# define cost and predictions
	cost = -(thT * T.log(thY)).sum() + reg * ((W1 * W1).sum() + (b1 * b1).sum() + (W2 * W2).sum() + (b2 * b2).sum())
	prediction = T.argmax(thY, axis=1)

	# define the update functions
	update_W1 = W1 - lr * T.grad(cost, W1)
	update_b1 = b1 - lr * T.grad(cost, b1)
	update_W2 = W2 - lr * T.grad(cost, W2)
	update_b2 = b2 - lr * T.grad(cost, b2)

	# define the train function
	train = theano.function(
		inputs=[thX, thT],
		updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)],
	)

	# define get prediction
	get_prediction = theano.function(
		inputs=[thX, thT],
		outputs=[cost, prediction]
	)

	LL = []
	for i in range(max_iter):
		for j in range(n_batches):
			Xbatch = Xtrain[j * batch_sz:(j * batch_sz + batch_sz), ]
			Ybatch = Ytrain_ind[j * batch_sz:(j * batch_sz + batch_sz), ]

			train(Xbatch, Ybatch)
			if j % print_period == 0:
				cost_val, prediction_val = get_prediction(Xtest, Ytest_ind)
				err = error_rate(prediction_val, Ytest)
				print("Cost / err at iteration i=%d, j=%d: %.3f / %.3f" % (i, j, cost_val, err))
				LL.append(cost_val)

	plt.plot(LL)
	plt.show()


if __name__ == "__main__":
	main()
