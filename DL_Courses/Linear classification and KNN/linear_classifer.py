import numpy as np


def softmax(predictions):
    '''
    Computes probabilities from scores

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output

    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    #normalized = predictions - np.max(predictions) Вычитает максимум из всех классов(т.е самый максимальный элемент в таблице, а мне нужен среди конкретного класса)
    normalized = predictions - np.max(predictions,axis=1).reshape(predictions.shape[0],1)
    fun_vectorize = np.vectorize(np.exp) #Создает объект, который позволяет применить функцию к каждому элементу массива
    exp_num = fun_vectorize(normalized) # exp_num - массив exp(normalized)
    sum_exp_num = np.sum(exp_num, axis=1).reshape(exp_num.shape[0],1) # вычисляет сумму элементов массива экспонент
    softmax_num = exp_num / sum_exp_num #Вычисляем функцию softmax
    return softmax_num
    raise Exception("Not implemented!")


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss

    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    
    
    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    logfunction_array = np.vectorize(np.log) 
    prob_class = logfunction_array(probs) #Делайем массив из логарифмов вероятностей
    real_probability = np.zeros(probs.shape,np.float64) # Создаем массив реальных значений вероятости
    demention = np.array([i for i in range(target_index.shape[0])])#Создаем массив от 0 до размера батча для того чтобы передвть его в real_probability с таргет индексом и в одно действие заменить нужные 0 на 1
    real_probability[demention,target_index.reshape(target_index.shape[0] * target_index.shape[1])] = 1
    cross_entropy_losses = np.sum(-prob_class * real_probability) / probs.shape[0] #Ищeм среднюю лос функцию только для 1 класса,из всех 
    return cross_entropy_losses

    raise Exception("Not implemented!")


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
    # TODO implement softmax with cross-entropy
    # Your final implementation shouldn't have any loops
    soft_max = softmax(predictions)
    loss = cross_entropy_loss(soft_max,target_index)
    dprediction = np.zeros(predictions.shape,np.float)
    real_probability = np.zeros(predictions.shape,np.float64)
    demention = np.array([i for i in range(target_index.shape[0])])
    target_index_shape = target_index.reshape(target_index.shape[0] * target_index.shape[1])
    real_probability[demention,target_index_shape] = 1
    dprediction = (-real_probability + soft_max) / predictions.shape[0]
    return loss, dprediction
    raise Exception("Not implemented!")




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

    # TODO: implement l2 regularization and gradient
    # Your final implementation shouldn't have any loops
    loss = reg_strength * np.sum(W ** 2)
    grad = 2 * reg_strength * W
    return loss, grad
    
    

def linear_softmax(X, W, target_index):
    '''
    Performs linear classification and returns loss and gradient over W

    Arguments:
      X, np array, shape (num_batch, num_features) - batch of images
      W, np array, shape (num_features, classes) - weights
      target_index, np array, shape (num_batch) - index of target classes

    Returns:
      loss, single value - cross-entropy loss
      gradient, np.array same shape as W - gradient of weight by loss

    '''
    predictions = np.dot(X, W) # Формируем массив предсказаний
    predictions = predictions - np.max(predictions, axis = 1).reshape(predictions.shape[0],1)# Нормируем МП
    exp_vectorize = np.vectorize(np.exp)# Задаём функцию, которая сможет применить експаненту ко всему масству
    predictions_exp = exp_vectorize(predictions)# Применяем
    softmax = predictions_exp / np.sum(predictions_exp, axis = 1).reshape(predictions_exp.shape[0],1)# находим функцию софтмакс#РЕШЕЙП ОБЯЗАТЕЛЕН.Без него сумма по строчкам не равна 1. Поэтому я Ебался 2 дня
    ln_vectorize = np.vectorize(np.log)
    ln_softmax = ln_vectorize(softmax)
    probability = np.zeros(predictions.shape)# Массив с нулевой вероятностью
    demention = np.array([i for i in range(target_index.shape[0])]) #размерность вектора, строки в которых вероятность с 0 будет заменена на 1
    probability[demention,target_index] = 1 #
    loss = np.sum(-probability * ln_softmax) / probability.shape[0]
    dW = np.dot(X.transpose(), softmax - probability) /  probability.shape[0]  #Ищем градиет, мы складываем  значения функции софтмас  с вероятностью и домнажемаем на ветор переменных Х.ПОСЛЕ МАНИПУЛЯЦИЙ СУММ. вЫХОДИТ ОБЫЧНЫЙ ВЕКТОР. ЕГО НУЖНО РЕШЕЙПНУТЬ.
    # TODO implement prediction and gradient over W
    # Your final implementation shouldn't have any loop
    return loss, dW

    raise Exception("Not implemented!")
    

class LinearSoftmaxClassifier():
    def __init__(self):
        self.W = None

    def fit(self, X, y, batch_size=100, learning_rate=1e-7, reg=1e-5,
            epochs=1):
        '''
        Trains linear classifier
        
        Arguments:
          X, np array (num_samples, num_features) - training data
          y, np array of int (num_samples) - labels
          batch_size, int - batch size to use
          learning_rate, float - learning rate for gradient descent
          reg, float - L2 regularization strength
          epochs, int - number of epochs
        '''

        num_train = X.shape[0]
        num_features = X.shape[1]
        num_classes = np.max(y)+1
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, num_classes)
        print(self.W.shape)
        loss_history = []
        for epoch in range(epochs):
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections) #Индексы тренировочных даннх, которые мы должны взять для тренировки
            loss, dW = 0, 0
            for i in batches_indices:#Пройдемся по всем батчам 30 раз
                X_batches = X[i]
                target_index = y[i]
                loss_entropy, dW_entropy = linear_softmax(X_batches, self.W, target_index)
                loss_reg_entropy, dW_reg_entropy = l2_regularization(self.W, reg)
                loss = loss + (loss_entropy + loss_reg_entropy)
                dW = dW + (dW_entropy + dW_reg_entropy)
            self.W = self.W - learning_rate * dW
            loss_history.append(loss)
            # TODO implement generating batches from indices
            # Compute loss and gradients
            # Apply gradient to weights using learning rate
            # Don't forget to add both cross-entropy loss
            # and regularization!

            # end
            print("Epoch %i, loss: %f" % (epoch, loss))

        return loss_history

    def predict(self, X):
        '''
        Produces classifier predictions on the set
       
        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        '''
        y_pred = np.zeros(X.shape[0], dtype=np.int)
        y_classes_numers = np.dot(X,self.W)
        y_bull = y_classes_numers == np.max(y_classes_numers,axis=1).reshape(X.shape[0],1)#Булевая маска, (максимальное значение Y)
        pred_array = np.zeros([X.shape[0],self.W.shape[1]],dtype=np.int)
        pred_weth = np.arange(self.W.shape[1])
        pred_array = pred_array + pred_weth
        y_pred = pred_array[y_bull]
       # y_pred =  # нужен индекс максимального
        # TODO Implement class prediction
        # Your final implementation shouldn't have any loops

        return y_pred



                
                                                          

            

                
