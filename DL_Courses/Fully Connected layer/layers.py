import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(W ** 2)
    grad = reg_strength * 2 * W
    
    return loss, grad


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """

    preds = preds - np.max(preds, axis = 1).reshape(preds.shape[0], 1)
    
    exp_vectorize = np.vectorize(np.exp)
    exp_preds = exp_vectorize(preds)
    
    soft_max = (exp_preds / np.sum(exp_preds, axis = 1).reshape(exp_preds.shape[0],1))
    
    ln_vectorize = np.vectorize(np.log)
    ln_softmax = ln_vectorize(soft_max)
    
    real_probability = np.zeros_like(preds)
    real_probability[np.arange(preds.shape[0]),target_index] = 1
    
    loss = np.sum(-real_probability * ln_softmax) / preds.shape[0]
    
    d_preds = (soft_max - real_probability) / preds.shape[0]
    
    # TODO: Copy from the previous assignment
   
    return loss, d_preds


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        self.X = X
     
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
       
        res = np.maximum(0,X)
        return res

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
     
        X = self.X 
        d_result = (self.X > 0) * d_out  #X[a1,a2], grad[dw1,dw2]
     
        return d_result
        raise Exception("Not implemented!")

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        W = self.W.value
        B = self.B.value
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = Param(X)
        out = np.dot(X,W) + B
        return out
        raise Exception("Not implemented!")

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment
       
        dX = np.dot(d_out, self.W.value.transpose())
        
        dW = np.dot(self.X.value.transpose(), d_out)
        
        dB = np.dot(np.ones((self.X.value.shape[0], 1)).transpose(), d_out)
        
        
        
        self.W.grad = dW
        self.B.grad = dB
        
        return dX
        raise Exception("Not implemented!")


    def params(self):
        return {'W': self.W, 'B': self.B}
