import numpy as np


def forward(X, W1, b1, W2, b2, use_rectifier=True):
	if use_rectifier:
		# rectifier
		Z = X.dot(W1) + b1
		Z[Z < 0] = 0
	else:
		Z = 1 / (1 + np.exp(-(X.dot(W1) + b1)))

	A = Z.dot(W2) + b2
	expA = np.exp(A)
	Y = expA / expA.sum(axis=1, keepdims=True)
	return Y, Z


def derivative_w2(Z, T, Y):
	return Z.T.dot(Y - T)


def derivative_b2(T, Y):
	return (Y - T).sum(axis=0)


def derivative_w1(X, Z, T, Y, W2, use_rectifier=True):
	if use_rectifier:
		return X.T.dot(((Y - T).dot(W2.T) * (Z > 0)))  # for relu

	return X.T.dot(((Y - T).dot(W2.T) * (Z * (1 - Z))))  # for sigmoid


def derivative_b1(Z, T, Y, W2, use_rectifier=True):
	if use_rectifier:
		return ((Y - T).dot(W2.T) * (Z > 0)).sum(axis=0)  # for relu

	return ((Y - T).dot(W2.T) * (Z * (1 - Z))).sum(axis=0)  # for sigmoid
