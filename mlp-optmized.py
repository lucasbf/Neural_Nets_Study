import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1.0-sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1.0 - x**2

def mlp(layers, activation='tanh'):
    if activation == 'sigmoid':
        activation_function = sigmoid
        activation_prime_function = sigmoid_prime
    elif activation == 'tanh':
        activation_function = tanh
        activation_prime_function = tanh_prime

    # Set weights
    weights = []
    # layers = [2,2,1]
    # range of weight values (-1,1)
    # input and hidden layers - random((2+1, 2+1)) : 3 x 3
    for i in range(1, len(layers) - 1):
        r = 2 * np.random.random((layers[i - 1] + 1, layers[i] + 1)) - 1
        weights.append(r)
    # output layer - random((2+1, 1)) : 3 x 1
    r = 2 * np.random.random((layers[i] + 1, layers[i + 1])) - 1
    weights.append(r)

    return dict(activation=activation_function, activation_prime=activation_prime_function, weights=weights)

def fit(mlp, X, y, learning_rate=0.2, epochs=1000):
    # Add column of ones to X
    # This is to add the bias unit to the input layer
    ones = np.atleast_2d(np.ones(X.shape[0]))
    X = np.concatenate((ones.T, X), axis=1)

    idx = range(X.shape[0])
    for k in range(epochs):
        np.random.shuffle(idx)
        for i in idx:
            a = [X[i]]

            for l in range(len(mlp['weights'])):
                dot_value = np.dot(a[l], mlp['weights'][l])
                activation = mlp['activation'](dot_value)
                a.append(activation)
            # output layer
            error = y[i] - a[-1]
            deltas = [error * mlp['activation_prime'](a[-1])]

            # we need to begin at the second to last layer
            # (a layer before the output layer)
            for l in range(len(a) - 2, 0, -1):
                deltas.append(deltas[-1].dot(mlp['weights'][l].T) * mlp['activation_prime'](a[l]))

            # reverse
            # [level3(output)->level2(hidden)]  => [level2(hidden)->level3(output)]
            deltas.reverse()

            # backpropagation
            # 1. Multiply its output delta and input activation
            #    to get the gradient of the weight.
            # 2. Subtract a ratio (percentage) of the gradient from the weight.
            for i in range(len(mlp['weights'])):
                layer = np.atleast_2d(a[i])
                delta = np.atleast_2d(deltas[i])
                mlp['weights'][i] += learning_rate * layer.T.dot(delta)

        if k % 50 == 0: print 'epochs:', k

def predict(mlp,X):
    a = np.append([1], X)
    for l in range(0, len(mlp['weights'])):
        a = mlp['activation'](np.dot(a, mlp['weights'][l]))
    return a

nn = mlp([2,2,1],activation='tanh')
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([0, 1, 1, 0])
fit(nn,X, y,epochs=2000)
for e in X:
    print(e,predict(nn,e))