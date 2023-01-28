import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        hidden_layer_size, int - number of neurons in the hidden layer Колличество нейронов в скрытом слове
        reg, float - L2 regularization strength
        """
        self.reg = reg
        self.Linear1 = FullyConnectedLayer(n_input, hidden_layer_size) # (3072,3)
        self.ReLu = ReLULayer()
        self.Linear2 = FullyConnectedLayer(hidden_layer_size, n_output)    #Выдаст 10 предказаний N_OUTPUT отвечает за количество выходов, т-есть за количество нейонов
        # 2 полносвязных слоя гвоорят о том, что у нас 2 параметра W1,W2, B1,B2
        # TODO Create necessary layers
        #raise Exception("Not implemented!")

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes Наш таргет индекс
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        # self.params() - дает нам параметры полносВзного слоя
        
        # Вначале идет полносвязный слой, затем слой функции активации,
        # В слое 3 нейрона 
        
        
        
        params = self.params()
        
        W1 = params['W1']
        B1 = params['B1']
        W2 = params['W2']
        B2 = params['B2']
        
        W1.grad = np.zeros_like(W1.value)
        B1.grad = np.zeros_like(B1.value)
        W2.grad = np.zeros_like(W2.value)
        B2.grad = np.zeros_like(B2.value)
        
        
        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        
        forward_Lauer1 = self.Linear1.forward(X)
        activationReLu = self.ReLu.forward(forward_Lauer1)
        forward_Lauer2 = self.Linear2.forward(activationReLu)
        
        loss, grad_Cross_entropy = softmax_with_cross_entropy(forward_Lauer2, y)
        
        backward_2 = self.Linear2.backward(grad_Cross_entropy)
        bacward_ReLu = self.ReLu.backward(backward_2)
        backward_1 = self.Linear1.backward(bacward_ReLu)
        
        # К лоссу прибавляем все регуляризации. ОТ W1,B1, W2, B2
        # raise Exception("Not implemented!")
        W1_loss, W1_grad = l2_regularization(W1.value, self.reg)
        B1_loss, B1_grad = l2_regularization(B1.value, self.reg)
        W2_loss, W2_grad = l2_regularization(W2.value, self.reg)
        B2_loss, B2_grad = l2_regularization(B2.value, self.reg)
        
        sum_loss = W1_loss + W2_loss + B2_loss + B1_loss
        loss = loss + sum_loss
        
        W1.grad += W1_grad
        B1.grad += B1_grad
        W2.grad += W2_grad
        B2.grad += B2_grad
        
        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        
        forward_Lauer1 = self.Linear1.forward(X)
        activationReLu = self.ReLu.forward(forward_Lauer1)
        forward_Lauer2 = self.Linear2.forward(activationReLu) # выдает массив размероv кол-во семплов ХХ классы(всего классов 10)
       
        vectorizeEXP = np.vectorize(np.exp)
        exp = vectorizeEXP(forward_Lauer2)
        probs = exp / np.sum(exp, axis = 1).reshape(exp.shape[0], 1)
        
        pred = np.argmax(probs, axis=1)
        
        return pred

    def params(self): # Инициализация параметров линейного слоя. По итогу сюда будут записываться градиенты
        result = {
        'W1': self.Linear1.params()['W'],
        'B1': self.Linear1.params()['B'],
        'W2': self.Linear2.params()['W'],
        'B2': self.Linear2.params()['B']}

        # TODO Implement aggregating all of the params

       # raise Exception("Not implemented!")

        return result
    def best_params(self)
