import numpy as np
from .DropoutLayer import DropoutLayer

class Model:
    def __init__(self, *layers):
        self.__layers = layers
        self.__epoch = 1

    def getEpoch(self):
        return self.__epoch
    
    def setEpoch(self, epoch):
        self.__epoch = epoch

    def getLayers(self):
        return self.__layers
    
    def setLayers(self, layers):
        self.__layers = layers

    def train(self, X, Y, epochs = 1):
        for _ in range(epochs):
            # forward pass
            h = X
            for i in range(len(self.__layers)-1):
                h = self.__layers[i].forward(h)

            # backward pass
            grad = self.__layers[-1].gradient(Y, h)
            for i in range(len(self.__layers)-2,0,-1):
                newgrad = self.__layers[i].backward(grad)
                if hasattr(self.__layers[i], 'learn'):
                    self.__layers[i].learn(grad, self.__epoch)
                grad = newgrad

            # update epoch
            self.__epoch += 1
        
    def eval(self, X, Y, metric):
        Yhat = X
        for i in range(len(self.__layers)-1):
            if not isinstance(self.__layers[i], DropoutLayer):
                Yhat = self.__layers[i].forward(Yhat)

        if metric == "RMSE":
            return np.sqrt(np.mean((Y - Yhat) ** 2))
        elif metric == "SMAPE":
            return np.mean(np.abs(Y - Yhat) / (np.abs(Y) + np.abs(Yhat)))