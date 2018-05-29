import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


#example of a picture 
# index = 34
# plt.imshow(train_set_x_orig[index])
# print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.") 
# plt.show()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Preprocessing of the training and testing examples
# Reshape the training and testing examples

train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T

print("train_set_x_flatten shape: {}".format(str(train_set_x_flatten.shape)))
print("train_set_y shape: {}".format(str(train_set_y.shape)))
print("test_set_x_flatten shape: {}".format(str(test_set_x_flatten.shape)))
print("test_set_y shape: {}".format(str(test_set_y.shape)))
print("sanity check after reshaping: {}".format(str(train_set_x_flatten[0:5, 0])))

train_set_x = train_set_x_flatten / 255 
test_set_x = test_set_x_flatten / 255

# Building parts of the algorithm.

def sigmoid(z):
	"""
	Compute the sigmoid of z
	
	Arguments:
		z {scalor or np.array} 

	Returns:
		z {scalor or np.array} -- sigmoid of z
	"""
	s = 1 / (1 + np.exp(-z))

	return s

def initialize_with_zeros(dim):
	"""
	this function creates a vector of zeros of shape (dim, 1) for w and initializes b to 0.
	
	Arguments:
		dim {int} -- size of the vector we want.	
	
	Returns:
		w -- initialized vector of shape (dim, 1)
		b -- initialized scalor (corresponds to the bias)
	"""

	w = np.zeros(shape=(dim, 1))
	b = 0

	assert(w.shape == (dim, 1))
	assert(isinstance(b, float) or isinstance(b, int))
	return w, b

# now that the parameters are initialized we can do the forward and backward propogation steps
# for learning the parameters

def propagate(w, b, X, Y):
	"""Implementing the cost function and its gradient for the propogation
	
	Arguments:
		w {numpy array} -- weights	
		b {scalor} -- bias
		X {np.array} -- data of size num_px * num_px * 3, number of examples
		Y {np.array} -- true "label vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

	Return:
		cost -- negative log-likelihood cost for logistic regression
		dw -- gradient of the loss with respect to w, thus same shape as w
		db -- gradient of the loss with respect to b, thus same shape as b
	"""
	m = X.shape[1]

	# Forward propogation from x to cost
	A = sigmoid(np.dot(w.T, X) + b)
	cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m

	# Backward propogation to find grad
	dw = (1 / m) * np.dot(X, (A - Y).T)
	db = np.sum(A - Y) / m

	assert(dw.shape == w.shape)
	assert(db.dtype == float)
	cost = np.squeeze(cost)
	assert(cost.shape == ())

	grads = {"dw": dw,
			"db": db}

	return grads, cost

# now we want to update the parameters using gradient descent
# now the goal is to learn w and b by minimizing the cost function J.

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = True):
	"""
	This function optimizes w and b by running a gradient descent algorithm
	
	Arguments:
		w {numpy array} -- weights, a numpy array of size (num_px * num_px * 3, 1)
		b {scalor} -- bias
		X {numpy array} -- data of shape (num_px * num_px * 3, number of examples)
		Y {numpy array} -- true label vector 
		num_iterations {scalor} -- number of iteration of the oprimization loop
		learning_rate {scalor} -- learning rate of the gradient descent update rule
	
	Keyword Arguments:
		print_cost {bool} -- bool (default: {True})
	
	Returns:
		params -- dictionary containing the weights w and bias b
		grad -- dictionary containing the gradients of the weights and bias w and b with respect to the cost function
		costs -- list of all the cost computed during the optimization process
	"""
	costs = []

	for i in range(num_iterations):
		grads, cost = propagate(w, b, X, Y)

		# Retrieve the derivatives from the grads
		dw = grads["dw"]
		db = grads["db"]

		# update rule 
		w = w - (learning_rate * dw)
		b = b - (learning_rate * db)

		# Record the costs 
		if i % 100 == 0: 
			costs.append(cost)

		# print the cost after every 100 iterations
		if print_cost and i % 100 == 0:
			print("Cost after iteration {0}: {1}".format(i, cost))

	params = {"w" : w, "b": b}
	grads = {"dw" : dw, "db": db}

	return params, grads, costs


def predict(w, b, X):
    '''
    Predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
    
    Arguments:
    w {numpy array} -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b {scalor} -- bias, a scalar
    X {numpy array} -- data of size (num_px * num_px * 3, number of examples)
    
    Returns:
    Y_prediction -- a numpy array (vector) containing all predictions (0/1) for the examples in X
    '''
    m = X.shape[1]
    Y_prediction = np.zeros((1,m))
    w = w.reshape(X.shape[0], 1)
    
    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    A = sigmoid(np.dot(w.T,X) + b)
    for i in range(A.shape[1]):
        
        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1            
    
    assert(Y_prediction.shape == (1, m))
    
    return Y_prediction


def model(X_train, Y_train, X_test, Y_test, num_iterations = 1000, learning_rate = 0.5, print_cost = True):
	"""Builds the logistic regression model

	Arguments:
	    X_train -- training set represented by a numpy array of shape (num_px * num_px * 3, m_train)
	    Y_train -- training labels represented by a numpy array (vector) of shape (1, m_train)
	    X_test -- test set represented by a numpy array of shape (num_px * num_px * 3, m_test)
	    Y_test -- test labels represented by a numpy array (vector) of shape (1, m_test)
	    num_iterations -- hyperparameter representing the number of iterations to optimize the parameters
	    learning_rate -- hyperparameter representing the learning rate used in the update rule of optimize()
	    print_cost -- Set to true to print the cost every 100 iterations
	    
	    Returns:
	    d -- dictionary containing information about the model.
	    """
	# initialize the parameters with zeros
	w, b = initialize_with_zeros(X_train.shape[0])

	# Gradient descent
	parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

	# Retrieve parameters w and b from dictionary "parameters"
	w = parameters["w"]
	b = parameters["b"]

	# predict the test/train set examples 
	Y_prediction_test = predict(w, b, X_test)
	Y_prediction_train = predict(w, b, X_train)

	# Print train/test Errors
	print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
	print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    
	d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}
    
	return d


if __name__ == '__main__':
	learning_rates = [0.01, 0.001, 0.0001]
	models = {}
	for i in learning_rates:
	    print ("learning rate is: " + str(i))
	    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 1500, learning_rate = i, print_cost = True)
	    print ('\n' + "-------------------------------------------------------" + '\n')

	for i in learning_rates:
	    plt.plot(np.squeeze(models[str(i)]["costs"]), label= str(models[str(i)]["learning_rate"]))

	plt.ylabel('cost')
	plt.xlabel('iterations (hundreds)')

	legend = plt.legend(loc='upper center', shadow=True)
	frame = legend.get_frame()
	frame.set_facecolor('0.90')
	plt.show()






