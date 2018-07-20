import numpy as np
import h5py
import matplotlib.pyplot as plt
from dnn_utils import sigmoid, sigmoid_backward, relu, relu_backward
from lr_utils import load_dataset

plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

np.random.seed(1)

# def initialize_parameters_deep(layer_dims):
#     """
#     Arguments:
#     layer_dims -- python array (list) containing the dimensions of each layer in our network
    
#     Returns:
#     parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
#                     Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
#                     bl -- bias vector of shape (layer_dims[l], 1)
#     """
    
#     np.random.seed(1)
#     parameters = {}
#     L = len(layer_dims)            # number of layers in the network

#     for l in range(1, L):
#         parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) #*0.01
#         parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
#         assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
#         assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

        
#     return parameters



def initialize_parameters_deep(L):
	"""initialize all the W's and b's
	
	Arguments:
		L {list} -- [list of all the dims in the network]
	"""
	np.random.seed(1)
	params = {}

	for l in range(1, len(L)):
		# * 0.01 -> makes the gradient closer to 0 
		params['W'+str(l)] = np.random.randn(L[l], L[l-1]) / np.sqrt(L[l-1])
		params['b'+str(l)] = np.zeros((L[l], 1)) 

		assert(params['W' + str(l)].shape == (L[l], L[l-1]))
		assert(params['b' + str(l)].shape == (L[l], 1))

	return params		


def linear_forward(A, W, b):
	"""linear part of forward prop
	
	Arguments:
		A {np array} -- [(l-1 x m)]
		W {np array} -- [(l x l-1)]
		b {np array} -- [(l x 1)]
	
	Returns:
		[Z ,cache] -- [return Z and A, W, B]
	"""

	Z = np.dot(W, A) + b
	
	assert(Z.shape == (W.shape[0], A.shape[1]))
	cache = (A, W, b)
	
	return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """activation part of forward prop

    Arguments:
    	A_prev {np array} -- [(l-1 x m)]
    	W {[np.array]} -- [(l x l-1)]
    	b {[np.array]} -- [(1 x b)]
    	activation {string} -- [relu or sigmoid]

    Returns:
    	[np.array] -- [A , (A_prev, W , b, Z)]
    """ 

    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z) 

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
	
    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, params):
	"""
	Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
	
	Arguments:
		X {np array} -- [input data (features x m)]
		parameters {[type]} -- [w1,b1 ... wn, bn]
	
	Returns:
		[Al , cache] -- [Al: last post activation vector, all previous caches]
	"""
	
	caches = []
	A = X
	L = len(params) // 2				 
	
	# Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
	for l in range(1, L):
		A_prev = A 
		A, cache = linear_activation_forward(A_prev, params['W'+str(l)], params['b'+str(l)], "relu")
		caches.append(cache)
	
	# Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
	AL, cache = linear_activation_forward(A, params['W'+str(L)], params['b'+str(L)], "sigmoid")
	caches.append(cache)
	
	assert(AL.shape == (1,X.shape[1]))
			
	return AL, caches


def compute_cost(AL, Y):
	"""
	Implement the cost function defined by logistic loss.
	
	Arguments:
		AL {np array} -- [label predictions]
		Y {np array} -- [real label vector]
	
	Returns:
		[scalor] -- [cost]
	"""

	m = Y.shape[1]

	# Compute loss from aL and y.
	cost = np.multiply(np.log(AL), Y) + np.multiply(np.log(1-AL),1-Y)
	cost = - np.sum(cost) / m

	cost = np.squeeze(cost)	  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
	assert(cost.shape == ())
	
	return cost


def linear_backward(dZ, cache):
	"""
	Implement the linear portion of backward propagation for a single layer (layer l)
	
	Arguments:
		dZ {np array} -- [gradient of the cost wrt Z]
		cache {tuple} -- [(A_prev, W, b)]
	
	Returns:
		[dA_prev] -- [gradient of the cost wrt the activation of the previous layer]
		[dW] -- []
		[db] -- []

	"""

	A_prev, W, b = cache
	m = A_prev.shape[1]

	dW = np.dot(dZ, A_prev.T) / m
	db = np.sum(dZ, axis=1, keepdims=True) / m
	dA_prev = np.dot(W.T, dZ)
	
	assert (dA_prev.shape == A_prev.shape)
	assert (dW.shape == W.shape)
	assert (db.shape == b.shape)
	
	return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
	"""
	Implement the backward propagation for the LINEAR->ACTIVATION layer.
	
	Arguments:
		dA {np array} -- [post-activation gradient for current layer l]
		cache {tuple} -- [(linear_cache, activation_cache)]
		activation {string} -- the activation to be used in this layer
	
	Returns:
		[dA_prev] -- [gradient of the cost wrt the activation of the previous layer]
		[dW] -- []
		[db] -- []
	"""
	linear_cache, activation_cache = cache
	
	if activation == "relu":
		dZ = relu_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
		
	elif activation == "sigmoid":
		dZ = sigmoid_backward(dA, activation_cache)
		dA_prev, dW, db = linear_backward(dZ, linear_cache)
	
	return dA_prev, dW, db


def L_model_backward(AL, Y, caches):
	"""
	Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
	
	Arguments:
		AL {np array } -- probability vector, output of the forward propagation (L_model_forward())
		Y {np array} -- true "label" vector
		caches {tuple} -- list of caches containing:
				every cache of linear_activation_forward() with "relu" (it's caches[l], for l in range(L-1) i.e l = 0...L-2)
				the cache of linear_activation_forward() with "sigmoid" (it's caches[L-1])
	
	Returns:
	grads -- A dictionary with the gradients
	"""

	grads = {}
	L = len(caches) # the number of layers
	m = AL.shape[1]
	Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
	
	# Initializing the backpropagation
	dAL = -( np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
	
	# Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "dAL, current_cache". Outputs: "grads["dAL-1"], grads["dWL"], grads["dbL"]
	current_cache = caches[L-1]
	grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
	
	# Loop from l=L-2 to l=0
	for l in reversed(range(L-1)):
		# lth layer: (RELU -> LINEAR) gradients.
		# Inputs: "grads["dA" + str(l + 1)], current_cache". Outputs: "grads["dA" + str(l)] , grads["dW" + str(l + 1)] , grads["db" + str(l + 1)] 
		current_cache = caches[l]
		
		dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
		grads["dA" + str(l)] = dA_prev_temp
		grads["dW" + str(l + 1)] = dW_temp
		grads["db" + str(l + 1)] = db_temp

	return grads


def update_parameters(parameters, grads, learning_rate):
    """Update parameters using gradient descent
    
    Arguments:
        params {dict} -- [Wl, bl]
        grads {dict} -- [dAl, dW1, db1]
        learning_rate {scalor} -- [convergence rate]
    
    Returns:
        [type] -- [description]
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    for l in range(L):
        parameters["W" + str(l+1)] -= learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] -= learning_rate * grads["db" + str(l+1)]
    return parameters


def L_layer_model(X, Y, layers_dim, learning_rate = 0.0075, num_iterations = 3000):
    """impliment a layred Neural Network: [LINEAR->RELU]*(L-1)->[LINEAR->SIGMOID]
    
    Arguments:
        X {np array} -- [the input array (feature x m)]
        Y {np array} -- [the list of truth values]
        layers_dim {list} -- [12288, 20, 7, 5, 1]
    
    Keyword Arguments:
        learning_rate {number} -- [description] (default: {0.0075})
        num_iterations {number} -- [description] (default: {3000})
    Return:
        params -- parameters learnt by the model.
    """

    np.random.seed(1)
    costs = []

    # Parameters initialization
    params = initialize_parameters_deep(layers_dim)

    #gradient descent 
    for i in range(0, num_iterations):
        # forward propogation
        AL, caches = L_model_forward(X, params)

        # compute cost
        cost = compute_cost(AL, Y)

        # back propogation
        grads = L_model_backward(AL, Y, caches)

        #upgrade params
        params = update_parameters(params, grads, learning_rate)

        if i%100 == 0:
            print("Cost after iteration %i %f" %(i, cost))
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return params


train_x_orig, train_y, test_x_orig, test_y, classes = load_dataset()

index = 4
plt.imshow(train_x_orig[index])
print("y = " + str(train_y[0, index]) + ". It is a " + classes[train_y[0, index]].decode('utf-8') + " picture")
plt.show()

m_train = train_x_orig.shape[0]
num_px = train_x_orig.shape[1]
m_test = test_x_orig.shape[0]

print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x_orig shape: " + str(train_x_orig.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x_orig shape: " + str(test_x_orig.shape))
print ("test_y shape: " + str(test_y.shape))

train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T / 255.
test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T / 255.

print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))


# L layer Neural Network
layers_dims = [12288, 20, 7, 5, 1]

parameters = L_layer_model(train_x, train_y, layers_dims, num_iterations = 2500)























