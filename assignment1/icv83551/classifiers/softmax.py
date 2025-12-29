from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in range(num_train):
        scores = X[i].dot(W)

        # compute the probabilities in numerically stable way
        scores -= np.max(scores)
        p = np.exp(scores)
        p /= p.sum()  # normalize
        logp = np.log(p)

        loss -= logp[y[i]]  # negative log probability is the loss
        
        # update the gradient
        for j in range(num_classes):
            if j == y[i]:
                dW[:,j] += (p[j]-1) * X[i]
            else:
                dW[:,j] += p[j] * X[i]

    # normalized hinge loss plus regularization
    loss = loss / num_train + reg * np.sum(W * W)
    
    dW = dW / num_train + 2*reg*W


    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    
    # number of samples
    N = X.shape[0]
    
    # compute scores
    scores = X.dot(W)
    
    # substract the max of every row for numerical stability
    row_max = np.max(scores, axis=1)
    row_max = row_max.reshape(N,1)
    scores = scores - row_max
    
    # compute the softmax
    exp_scores = np.exp(scores)
    row_sum = np.sum(exp_scores, axis=1)
    row_sum = row_sum.reshape(N,1)
    prob = exp_scores / row_sum
    
    # compute the loss
    probs_correct_class = prob[np.arange(N),y]
    loss = -np.sum(np.log(probs_correct_class)) / N
    
    # regularization for the loss
    loss += reg * np.sum(W*W)
    
    # compute gradient use the probs from up
    d_scores = prob.copy()
    d_scores[np.arange(N),y] = d_scores[np.arange(N),y] -1
    d_scores = d_scores / N
    
    dW = X.T.dot(d_scores)
    
    # regularization for the gradient
    dW += 2*reg*W

    return loss, dW
