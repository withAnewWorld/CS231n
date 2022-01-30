from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        
        W1 = np.random.normal(0, scale=weight_scale, size=(input_dim, hidden_dim))
        b1 = np.zeros(hidden_dim)
        W2 = np.random.normal(0, scale=weight_scale, size=(hidden_dim, num_classes))
        b2=np.zeros(num_classes)
        self.params['W1']= W1
        self.params['W2']= W2
        self.params['b1']= b1
        self.params['b2']= b2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
      

        X=X.reshape(X.shape[0],-1)
        result_0 = X.dot(self.params['W1'])+self.params['b1']
        result_1=np.maximum(0, result_0)
        result_2=result_1.dot(self.params['W2'])+self.params['b2']
        scores = result_2

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        N = X.shape[0]
        D = self.params['W1'].shape[0]
        H = self.params['W1'].shape[1]
        C = self.params['W2'].shape[1]
      
        
        scores -= np.max(scores, axis=1, keepdims=True)
        exp_scores= np.exp(scores)
        sum_exp_scores = np.sum(exp_scores, axis=1, keepdims=True)
        loss_i = exp_scores[range(N), y].reshape(N, -1)/sum_exp_scores
        loss_i = -np.log(loss_i)
        loss = np.sum(loss_i)
        loss /= X.shape[0]
        loss += (self.reg*0.5)*np.sum(self.params['W1']*self.params['W1'])
        loss += (self.reg*0.5)*np.sum(self.params['W2']*self.params['W2']) 

        dsmx = exp_scores/np.tile(sum_exp_scores, (1, scores.shape[1]))
        dsmx[range(dsmx.shape[0]), y] -= 1
        dsmx /= N # dsmx.shape = N*C

        ones_b2=np.ones(N)
        db2= dsmx.T.dot(ones_b2) # C = C*N.dot(N*1)

        dW2= result_1.T.dot(dsmx) # H*C = N*H.dot(N*C)
        dW2+= self.reg*self.params['W2'] #dW2.shape = H*C

        dwxb_1= dsmx.dot(self.params['W2'].T)
        dwxb_1[result_1<=0]=0
        #result_1 = max(0, w1x+b1)
        # W1X+b1.shape = N*H
        # N*H = N*C.dot(C*H)

        ones_b1= np.ones(N)
        db1=dwxb_1.T.dot(ones_b1) # H = H*N.dot(N) 

        dwx_1=dwxb_1 

        dW1=X.T.dot(dwx_1) # D*H = D*N.dot(N*H)
        dW1+= self.reg*self.params['W1']

        


        grads['b2']=db2
        grads['b1']=db1
        grads['W2']=dW2
        grads['W1']=dW1
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
