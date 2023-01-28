import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    raise Exception("Not implemented!")

    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    '''
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
    '''
    exp_vectorized = np.vectorize(np.exp)
    all_exp = exp_vectorized(predictions)
    
    softmax = all_exp / np.sum(all_exp,axis=1).reshape(-1,1)
    
    ln_vectorized = np.vectorize(np.log)
    ln_softmax = ln_vectorized(softmax)
    
    
    predictions_true = np.zeros_like(predictions)
    predictions_true[np.arange(predictions.shape[0]), target_index] = 1
    
    loss = np.sum(-predictions_true * ln_softmax) / predictions.shape[0]
    
    dprediction = (-predictions_true + softmax) / predictions.shape[0]
    
    return loss, dprediction
    


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        return np.maximum(X, 0)
        

    def backward(self, d_out):
        # TODO copy from the previous assignment
        X = self.X
        d_X = X > 0
        
        d_result = d_X * d_out
        
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        W = self.W.value
        B = self.B.value
        self.X = Param(X)
        return np.dot(X,W) + B
        
        raise Exception("Not implemented!")

    def backward(self, d_out):
        # TODO copy from the previous assignment
        #input только по Х.
        
        # d_out - (batch_size, n_output)
        
        # X - (batch_size, n_input)
        
        # W - (n_input, n_output)
        X = self.X.value
        W = self.W.value
        B = self.B.value
        
        self.W.grad = np.dot(X.transpose(), d_out)
        self.B.grad = np.dot(np.ones((X.shape[0],1)).transpose(), d_out)
     
        d_input = np.dot(d_out, W.transpose())
       
        return d_input

    def params(self):
        return { 'W': self.W, 'B': self.B }

    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels #Сколько у нас входных каналов 
        out_channels, int - number of output channels # Сколько выходных каналов
        filter_size, int - size of the conv filter # Размер фильтра
        padding, int - number of 'pixels' to pad on each side #        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))

        self.padding = padding


    def forward(self, X):
        
        padding = self.padding
        
        if padding != 0:
            X = np.insert(X, 0, np.zeros([padding]), axis = 2)
            X = np.insert(X, X.shape[2], np.zeros([padding]), axis = 2)
            X = np.insert(X, 0, np.zeros([padding]), axis = 1)
            X = np.insert(X, X.shape[1], np.zeros([padding]), axis = 1)
       
          
        
        batch_size, height, width, channels = X.shape
        
        out_height = height - self.filter_size + 1 #Нужная высота Кол-во X по высоте - фильтр + 1 
        out_width =  width - self.filter_size + 1 #Нужная ширина
        
        # За глубиной следить не надо.
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        self.X = X
        filter_size = self.filter_size
        out_channels = self.out_channels
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        W_reshaped = self.W.value.reshape(-1, out_channels)
        B = self.B.value
         #Решейпим веса, в 2-мерое п-во 
        
       
        result = np.zeros([batch_size, out_height, out_width, out_channels])
        
        for y in range(out_height):
            for x in range(out_width):
                X_without_reshape = X[:, y:y+filter_size, x:x+filter_size, :]
                X_reshaped = X_without_reshape.reshape(batch_size, -1)
                result[:, y, x, :] = np.dot(X_reshaped, W_reshaped) + B
               
        return result



    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients
        # Инициализируем Х
        X = self.X
        filter_size = self.filter_size
        padding = self.padding
        
        #Инициализируем градиенты.
        self.W.grad = np.zeros_like(self.W.value)
        self.B.grad = np.zeros_like(self.B.value)
        
        batch_size, height, width, channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        #Инициализируем веса.
        W_reshaped = self.W.value.reshape(-1, out_channels)
        
        dX = np.zeros_like(X)
        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output
        # Try to avoid having any other loops here too
       
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                true_grad = d_out[:, y, x, :] # Внутрянняя размерность d_out(высота и ширина) такая же как и у result в слое forward
                
                X_without_reshape = X[:, y:y+filter_size, x:x+filter_size, :]
                X_reshaped = X_without_reshape.reshape((batch_size, -1))
                
                dW = np.dot(X_reshaped.transpose(), true_grad)
                dW_reshaped = dW.reshape((filter_size, filter_size, self.in_channels, out_channels))
                
                dB = np.dot(np.ones((batch_size,)).transpose(), true_grad)
                
                self.W.grad += dW_reshaped
                self.B.grad += dB
                
                dX_not_res = np.dot(true_grad, W_reshaped.transpose())
                dX_res = dX_not_res.reshape((batch_size, filter_size, filter_size, self.in_channels)) #Решейпим по фильтер сайз, тк uhfдиенты считаем по области фильтра. ТОЧНО
                
                dX[:, y:y+filter_size, x:x+filter_size,:] += dX_res
                
        dX = dX[:, padding:height - padding, padding: width-padding,:]        
                
        return dX
       

    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        pool_size = self.pool_size
        stride = self.stride
        self.X = X
        #Проверить на чётность размер матрицы и посмотреть до какого элемента может идти макс пулл с шагом 
        #делить на размер окна в общем случае это пойдет
        out_height = int(np.floor((height - pool_size) / stride)) + 1
        out_width = int(np.floor((width - pool_size) / stride)) + 1
            
        max_polling = np.zeros((batch_size, out_height, out_width, channels))
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
       
        for batch in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    for channel in range(channels): # В каждом батче и каждом канале слой maxpooling  свой
                        y_source = y * stride # Таким образом мы по примерам, с шагом страйд
                        x_source = x * stride
                        pool = X[batch, y_source:y_source+pool_size, x_source:x_source+pool_size, channel] # выбираем элементы матрицы размером maxpool
                        maximum = np.max(pool)
                        max_polling[batch, y, x, channel] = maximum
                
      
        return max_polling         
        

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        pool_size = self.pool_size
        stride = self.stride
        X = self.X
        batch_size, height, width, in_channels = X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        d_X = np.zeros_like(X)
        
        for batch in range(batch_size):
            for y in range(out_height):
                for x in range(out_width):
                    for channel in range(in_channels):
                        mask = np.zeros((pool_size, pool_size))
                        y_source = y * stride
                        x_source = x * stride
                        pool = X[batch,
                                 y_source:np.minimum(y_source+pool_size, height),
                                 x_source:np.minimum(x_source+pool_size, width), channel]
                        
                        maximum = np.max(pool)
                        max_count = np.count_nonzero(pool == maximum)
                        argmax = np.argwhere(pool==maximum) #Находит индексы элементов, которые равны максимуму. Почему-то пишет в жвумерный массив.Поэтому нужно непосредсвенно извлекать индксы: argmax[:,0], argmax[:,1] 
                       
                        mask[argmax[:,0], argmax[:,1]] = d_out[batch, y, x, channel] / max_count
                        
                        d_X[batch, y_source:y_source+pool_size, x_source:x_source+pool_size, channel] += mask
                        
        return d_X

    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        return X.reshape((batch_size, height*width*channels))

    def backward(self, d_out):
        # TODO: Implement backward pass
        batch_size, height, width, channels = self.X.shape
        
        return d_out.reshape((batch_size, height, width, channels))
     

    def params(self):
        # No params!
        return {}
