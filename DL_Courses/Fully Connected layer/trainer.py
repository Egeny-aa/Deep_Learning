from copy import deepcopy # Ху из ит

import numpy as np
from metrics import multiclass_accuracy


class Dataset:
    """
    Utility class to hold training and validation data
    """

    def __init__(self, train_X, train_y, val_X, val_y):
        self.train_X = train_X
        self.train_y = train_y
        self.val_X = val_X
        self.val_y = val_y


class Trainer:
    """
    Trainer of the neural network models
    Perform mini-batch SGD with the specified data, model,
    training parameters and optimization rule
    """

    def __init__(self, model, dataset, optim,
                 num_epochs=20,
                 batch_size=20,
                 learning_rate=1e-2,
                 learning_rate_decay=1.0):
        """
        Initializes the trainer

        Arguments:
        model - neural network model
        dataset, instance of Dataset class - data to train on
        optim - optimization method (see optim.py)
        num_epochs, int - number of epochs to train
        batch_size, int - batch size
        learning_rate, float - initial learning rate
        learning_rate_decal, float - ratio for decaying learning rate
           every epoch
        """
        self.dataset = dataset
        self.model = model
        self.optim = optim
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.learning_rate_decay = learning_rate_decay

        self.optimizers = None

    def setup_optimizers(self): # оптимизирует параметры
        params = self.model.params()
        self.optimizers = {}
        for param_name, param in params.items():
            self.optimizers[param_name] = deepcopy(self.optim)

    def compute_accuracy(self, X, y):
        """
        Computes accuracy on provided data using mini-batches
        """
        indices = np.arange(X.shape[0])
        sections = np.arange(self.batch_size, X.shape[0], self.batch_size)
        batches_indices = np.array_split(indices, sections)

        pred = np.zeros_like(y)

        for batch_indices in batches_indices:
            batch_X = X[batch_indices]
            pred_batch = self.model.predict(batch_X)
            pred[batch_indices] = pred_batch

        return multiclass_accuracy(pred, y)

    def fit(self):
        """
        Trains a model
        """
        if self.optimizers is None:
            self.setup_optimizers()

        num_train = self.dataset.train_X.shape[0] 

        loss_history = []
        train_acc_history = []
        val_acc_history = []
        
        # инициализация данных
        X = self.dataset.train_X
        y = self.dataset.train_y
        
        for epoch in range(self.num_epochs):
            shuffled_indices = np.arange(num_train) # массив длиной от 0 до len(num_train)
            
            np.random.shuffle(shuffled_indices) # мешаем индексы эпохи
            
            sections = np.arange(self.batch_size, num_train, self.batch_size) # Делает массив с шаго в длину батча
            
            batches_indices = np.array_split(shuffled_indices, sections) # формирует массив индексов батчей
            
        
            batch_losses = []
            
            

            for batch_indices in batches_indices:
                # TODO Generate batches based on batch_indices and
                # use model to generate loss and gradients for all
                # the params
                 
                data_X = X[batch_indices] # выбираем из данных только те, которые по батч индексу(рандомные перемешанные)
                data_y = y[batch_indices]
                
                Loss = self.model.compute_loss_and_gradients(data_X, data_y) # Тренируем модель на этих данных
                
                           
                for param_name, param in self.model.params().items():
                    optimizer = self.optimizers[param_name]
                    param.value = optimizer.update(param.value, param.grad, self.learning_rate)
               
                batch_losses.append(Loss)#Записываем лосс в масив лоссов по батчу

            if np.not_equal(self.learning_rate_decay, 1.0): # выводит Тру, если значения передаваемые ему не равны друг-другу
                # TODO: Implement learning rate decay
                self.learning_rate = self.learning_rate * self.learning_rate_decay
                
                

            ave_loss = np.mean(batch_losses)

            train_accuracy = self.compute_accuracy(self.dataset.train_X,
                                                   self.dataset.train_y)

            val_accuracy = self.compute_accuracy(self.dataset.val_X,
                                                 self.dataset.val_y)

            print("Loss: %f, Train accuracy: %f, val accuracy: %f" %
                  (batch_losses[-1], train_accuracy, val_accuracy))

            loss_history.append(ave_loss)
            train_acc_history.append(train_accuracy)
            val_acc_history.append(val_accuracy)

        return loss_history, train_acc_history, val_acc_history
    
    def trainer_params(self):
        best_giper_params = {'batch_size':self.batch_size, 'learning_rate':self.learning_rate,'num_epochs':self.num_epochs, 'learning_rate_decay':self.learning_rate_decay}
        
        return best_giper_params
