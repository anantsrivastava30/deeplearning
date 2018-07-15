import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from snn_utils import plot_decision_boundary, load_planar_dataset, sigmoid

np.random.seed(1)


def layer_sizes(X, Y):
    """
    :param X :
    X - array of shape (input size, number of examples)
    :param Y:
    Y - lable of shape (output size, number of examples)
    :return:
    n_x - input layer
    n_h - hidden layer
    n_y - output layer
    """
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y


def initialize_parameters(n_x, n_h, n_y):
    """
    :param n_x:
    :param n_h:
    :param n_y:
    :return:
    params - dict of weights and biases W1, b1, W2, b2
    """
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}
    return params


def forward_propagation(X, params):
    """

    :param X:
    :param params:
    :return:
    A2 - the sigmoid of the last layer
    cache - dict containing Z1, A1, Z2, A2
    """
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    assert(A2.shape == (1, X.shape[1]))

    return A2, cache


def nn_cost(A2, Y, parameters):
    """
    compute the logistic loss
    :param parameters:
    :param A2:
    A2 - the sigmoid output of the forward propagation
    :param Y:
    Y - labels of shape (1, number of training examples
    :return:
    loss -
    """
    m = Y.shape[1]
    J = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), 1-Y)
    loss = - np.sum(J)/m
    loss = np.squeeze(loss)
    assert(isinstance(loss, float))

    return loss


def backward_propagation(params, cache, X, Y):
    """
    backward propagation ...
    :param params:
    params - W1, b1, W2, b2
    :param cache:
    cache - Z1, A1, Z2, A2
    :param X:
    X - input data of shape (2, number of examples)
    :param Y:
    Y - label vector of shape (1, number of examples)
    :return:
    grads - gradients of the weights
    """
    m = X.shape[1]

    W1 = params["W1"]
    W2 = params["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = np.dot(dZ2, A1.T) / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}

    return grads


def update_parameters(params, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule given above

    Arguments:
    parameters -- python dictionary containing parameters
    grads -- python dictionary containing gradients

    Returns:
    parameters -- python dictionary containing updated parameters
    """
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2

    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2}

    return params


def nn_model(X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    params = initialize_parameters(n_x, n_h, n_y)
    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        A2, cache = forward_propagation(X, params)

        cost = nn_cost(A2, Y, params)

        grads = backward_propagation(params, cache, X, Y)

        params = update_parameters(params, grads)

        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost))

    return params


def predict(params, X):
    """
    using the parameters trained find out the predictions
    :param params:
    :param X:
    :return:
    predictions - array of binary predictions
    """
    A2, cache = forward_propagation(X, params)
    predictions = np.asarray([0 if x<0.5 else 1 for x in A2.flatten()])

    return predictions


X, Y = load_planar_dataset()

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y, s=40)
plt.show()

shape_X = X.shape
shape_Y = Y.shape

m = shape_X[1]  # Training set size

print('The shape of X is: ' + str(shape_X))
print('The shape of Y is: ' + str(shape_Y))
print('I have m = %d training examples!' % m)

# parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
# plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
# plt.title("Decision Boundary for hidden layer size " + str(4))
# plt.show()

# predictions = predict(parameters, X)
# print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + '%')

N = 200
noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2,
                                                              n_classes=2, shuffle=True, random_state=None)
no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "gaussian_quantiles": gaussian_quantiles}

for dataset in datasets.values():
    X, Y = dataset
    X, Y = X.T, Y.reshape(1, Y.shape[0])

    # make blobs binary
    if dataset == "blobs":
        Y = Y % 2

    # Visualize the data
    plt.scatter(X[0, :], X[1, :], c=Y, s=40)
    plt.show()
    parameters = nn_model(X, Y, n_h=4, num_iterations=10000, print_cost=True)
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show()

    predictions = predict(parameters, X)
    print('Accuracy: %d' % float((np.dot(Y, predictions.T) + np.dot(1-Y, 1-predictions.T))/float(Y.size)*100) + '%')