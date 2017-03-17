import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression


class MNIST(object):
	def __init__(self):
		self.dataTestPath = 'Data/MNIST'
		self.dataTrainPath = 'Data/MNIST'

	def get_transformed_data(self, load_train=True, up_one_level=True):
		print("Reading in and transforming data...")
		df = pd.read_csv(self.get_file_path(load_train, up_one_level))
		data = df.as_matrix().astype(np.float32)
		np.random.shuffle(data)

		X = data[:, 1:]
		mu = X.mean(axis=0)
		X = X - mu  # center the data
		pca = PCA()
		Z = pca.fit_transform(X)
		Y = data[:, 0]

		plot_cumulative_variance(pca)

		return Z, Y, pca, mu

	def get_normalized_data(self, load_train=True, up_one_level=True):
		print("Reading in and transforming data...")

		df = pd.read_csv(self.get_file_path(load_train, up_one_level))
		data = df.as_matrix().astype(np.float32)
		np.random.shuffle(data)
		X = data[:, 1:]
		mu = X.mean(axis=0)
		std = X.std(axis=0)
		np.place(std, std == 0, 1)
		X = (X - mu) / std  # normalize the data
		Y = data[:, 0]
		return X, Y

	def get_file_path(self, load_train, up_one_level):
		pre = ''
		if up_one_level:
			pre = '../'
		if load_train:
			return pre + 'Data/MNIST/train.csv'
		return pre + 'Data/MNIST/test.csv'


class FaceRecognizer(object):
	def __init__(self):
		self.label_map = ['Anger', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

	def getData(self, balance_ones=True):
		# images are 48x48 = 2304 size vectors
		# N = 35887
		X = []
		Y = []
		first = True

		for line in open('../Data/fer2013/fer2013.csv'):
			if first:
				first = False
			else:
				row = line.split(',')
				Y.append(int(row[0]))
				X.append([int(p) for p in row[1].split()])

		X, Y = np.array(X) / 255.0, np.array(Y)

		if balance_ones:
			# balance the 1 class
			X0, Y0 = X[Y != 1, :], Y[Y != 1]
			X1 = X[Y == 1, :]
			X1 = np.repeat(X1, 9, axis=0)
			X = np.vstack([X0, X1])
			Y = np.concatenate((Y0, [1] * len(X1)))

		return X, Y

	def getImageData(self):
		X, Y = self.getData()
		N, D = X.shape
		d = int(np.sqrt(D))
		X = X.rehape(N, 1, d, d)

		return X, Y

	def getBinaryData(self):
		X = []
		Y = []
		first = True
		for line in open('../Data/fer2013/fer2013.csv'):
			if first:
				first = False
			else:
				row = line.split(',')
				y = int(row[0])
				if y == 0 or y == 1:
					Y.append(y)
					X.append([int(p) for p in row[1].split()])
		return np.array(X) / 255.0, np.array(Y)

	def showImages(self, balance_ones=True):
		X, Y = self.getData(balance_ones=balance_ones)

		while True:
			for i in range(7):
				x, y = X[Y == i], Y[Y == i]
				N = len(y)
				j = np.random.choice(N)

				plt.imshow(x[j].reshape(48, 48), cmap='gray')
				plt.title(self.label_map[y[j]])
				plt.show()
			prompt = input('Quit? Enter Y:\n')
			if prompt == 'Y':
				break;


def init_weight_and_biases(M1, M2):
	W = np.random.randn(M1, M2) / np.sqrt(M1 + M2)
	b = np.zeros(M2)
	return W.astype(np.float32), b.astype(np.float32)


def relu(x):
	return x * (x > 0)


def plot_cumulative_variance(pca):
	P = []
	for p in pca.explained_variance_ratio_:
		if len(P) == 0:
			P.append(p)
		else:
			P.append(p + P[-1])
	plt.plot(P)
	plt.show()
	return P


def forward(X, W, b):
	# softmax
	a = X.dot(W) + b
	# print "any nan in X?:", np.any(np.isnan(X))
	# print "any nan in W?:", np.any(np.isnan(W))
	# print "W:", W
	# print "X.dot(W):", X.dot(W)
	# print "b:", b
	# print "a:", a
	expa = np.exp(a)
	# print "expa:", expa
	y = expa / expa.sum(axis=1, keepdims=True)
	# exit()
	return y


def predict(p_y):
	return np.argmax(p_y, axis=1)


def error_rate(p_y, t):
	prediction = predict(p_y)
	return np.mean(prediction != t)


def cost(p_y, t):
	# print "any nan in log p_y?:", np.any(np.isnan(np.log(p_y)))
	# print "log(p_y):", np.log(p_y)
	# exit()
	tot = t * np.log(p_y)
	return -tot.sum()


def gradW(t, y, X):
	return X.T.dot(t - y)


def gradb(t, y):
	return (t - y).sum(axis=0)


def y2indicator(y):
	N = len(y)
	K = len(set(y))
	ind = np.zeros((N, K))
	for i in range(N):
		ind[i, y[i]] = 1
	return ind


def benchmark_full():
	model = MNIST()
	X, Y = model.get_normalized_data(up_one_level=False)

	print("Performing logistic regression...")
	# lr = LogisticRegression(solver='lbfgs')

	# # test on the last 1000 points
	# lr.fit(X[:-1000, :200], Y[:-1000]) # use only first 200 dimensions
	# print lr.score(X[-1000:, :200], Y[-1000:])
	# print "X:", X

	# normalize X first
	# mu = X.mean(axis=0)
	# std = X.std(axis=0)
	# X = (X - mu) / std

	Xtrain = X[:-1000, ]
	Ytrain = Y[:-1000]
	Xtest = X[-1000:, ]
	Ytest = Y[-1000:]

	# convert Ytrain and Ytest to (N x K) matrices of indicator variables
	N, D = Xtrain.shape
	Ytrain_ind = y2indicator(Ytrain)
	Ytest_ind = y2indicator(Ytest)

	W = np.random.randn(D, 10) / 28
	b = np.zeros(10)
	LL = []
	LLtest = []
	CRtest = []

	# reg = 1
	# learning rate 0.0001 is too high, 0.00005 is also too high
	# 0.00003 / 2000 iterations => 0.363 error, -7630 cost
	# 0.00004 / 1000 iterations => 0.295 error, -7902 cost
	# 0.00004 / 2000 iterations => 0.321 error, -7528 cost

	# reg = 0.1, still around 0.31 error
	# reg = 0.01, still around 0.31 error
	lr = 0.00004
	reg = 0.01
	for i in range(500):
		p_y = forward(Xtrain, W, b)
		# print "p_y:", p_y
		ll = cost(p_y, Ytrain_ind)
		LL.append(ll)

		p_y_test = forward(Xtest, W, b)
		lltest = cost(p_y_test, Ytest_ind)
		LLtest.append(lltest)

		err = error_rate(p_y_test, Ytest)
		CRtest.append(err)

		W += lr * (gradW(Ytrain_ind, p_y, Xtrain) - reg * W)
		b += lr * (gradb(Ytrain_ind, p_y) - reg * b)
		if i % 10 == 0:
			print("Cost at iteration %d: %.6f" % (i, ll))
			print("Error rate:", err)

	p_y = forward(Xtest, W, b)
	print("Final error rate:", error_rate(p_y, Ytest))
	iters = range(len(LL))
	plt.plot(iters, LL, iters, LLtest)
	plt.show()
	plt.plot(CRtest)
	plt.show()


def benchmark_pca():
	model = MNIST()
	X, Y, _, _ = model.get_transformed_data(up_one_level=False)
	X = X[:, :300]

	# normalize X first
	mu = X.mean(axis=0)
	std = X.std(axis=0)
	X = (X - mu) / std

	print("Performing logistic regression...")
	Xtrain = X[:-1000, ]
	Ytrain = Y[:-1000].astype(np.int32)
	Xtest = X[-1000:, ]
	Ytest = Y[-1000:].astype(np.int32)

	N, D = Xtrain.shape
	Ytrain_ind = np.zeros((N, 10))
	for i in range(N):
		Ytrain_ind[i, Ytrain[i]] = 1

	Ntest = len(Ytest)
	Ytest_ind = np.zeros((Ntest, 10))
	for i in range(Ntest):
		Ytest_ind[i, Ytest[i]] = 1

	W = np.random.randn(D, 10) / 28
	b = np.zeros(10)
	LL = []
	LLtest = []
	CRtest = []

	# D = 300 -> error = 0.07
	lr = 0.0001
	reg = 0.01
	for i in range(200):
		p_y = forward(Xtrain, W, b)
		# print "p_y:", p_y
		ll = cost(p_y, Ytrain_ind)
		LL.append(ll)

		p_y_test = forward(Xtest, W, b)
		lltest = cost(p_y_test, Ytest_ind)
		LLtest.append(lltest)

		err = error_rate(p_y_test, Ytest)
		CRtest.append(err)

		W += lr * (gradW(Ytrain_ind, p_y, Xtrain) - reg * W)
		b += lr * (gradb(Ytrain_ind, p_y) - reg * b)
		if i % 10 == 0:
			print("Cost at iteration %d: %.6f" % (i, ll))
			print("Error rate:", err)

	p_y = forward(Xtest, W, b)
	print("Final error rate:", error_rate(p_y, Ytest))
	iters = range(len(LL))
	plt.plot(iters, LL, iters, LLtest)
	plt.show()
	plt.plot(CRtest)
	plt.show()


if __name__ == '__main__':
	# benchmark_pca()
	benchmark_full()
