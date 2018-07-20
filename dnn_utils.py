import numpy as np

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
