import numpy as np
import matplotlib.pyplot as plt

def sigmoid(Z):
	"""sigmoid 
	
	Arguments:
		Z {[np.array]} -- [Wx + b]
	
	Returns:
		A - [np.array] -- [1 / 1+exp(- Wx + b)]
		cache - Z
	"""
	A = 1/(1+np.exp(-Z))
	cache = Z

	return A, cache

def relu(Z):
	"""rectified linear unit 
	
	Arguments:
		Z {[np.array]} -- [Wx + b]

	Returns:
		A - [np.array] -- [max(0,Z)]
		cache - Z
	"""
	A = np.maximum(0,Z)

	assert(A.shape == Z.shape)

	cache = Z
	return A, cache

def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
		dA - the activated gradient
		cache - Z

    Returns:
    	dZ - Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # for z <= 0, set dz to 0 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA - the acitvated gradient
    cache - Z

    Returns:
    	dZ - Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ


def print_mislabeled_images(classes, X, y, p):
    """
    Plots images where predictions and truth were different.
    X -- dataset
    y -- true labels
    p -- predictions
    """
    a = p + y
    mislabeled_indices = np.asarray(np.where(a == 1))
    print(mislabeled_indices)
    num_images = len(mislabeled_indices[0])
    for i in range(num_images):
        index = mislabeled_indices[1][i]
        
        plt.subplot(2, num_images, i + 1)
        plt.imshow(X[:,index].reshape(64,64,3))
        plt.title("Prediction: " + classes[int(p[0,index])].decode("utf-8") + " \n Class: " + classes[y[0,index]].decode("utf-8"))
        plt.show()