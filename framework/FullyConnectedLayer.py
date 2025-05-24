from .Layer import Layer

from .LinearLayer import LinearLayer
from .ReLULayer import ReLULayer

import numpy as np

class FullyConnectedLayer(Layer):

    def __init__(self, sizeIn, sizeOut, activationFunction):
        super().__init__()

        # initialize weights and biases
        # standard initialization
        if activationFunction == LinearLayer:
            range = 1e-4
            self.__weights = np.random.uniform(low = -range, high = range, size = (sizeIn, sizeOut))
            self.__biases = np.random.uniform(low = -range, high = range, size = (1, sizeOut))
        # he initialization
        elif activationFunction == ReLULayer:
            std = np.sqrt(2 / sizeIn)
            self.__weights = np.random.normal(loc = 0, scale = std, size = (sizeIn, sizeOut))
            self.__biases = np.zeros((1, sizeOut))
        
        # initialize ADAM terms
        self.__sW = np.zeros_like(self.__weights)
        self.__rW = np.zeros_like(self.__weights)
        self.__sb = np.zeros_like(self.__biases)
        self.__rb = np.zeros_like(self.__biases)

    def getWeights(self):
        return self.__weights
    
    def setWeights(self, weights):
        self.__weights = weights
    
    def getBiases(self):
        return self.__biases
    
    def setBiases(self, biases):
        self.__biases = biases
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = dataIn @ self.__weights + self.__biases
        self.setPrevOut(y)
        return y
    
    def gradient(self):
        return self.__weights.T
    
    def backward(self, gradIn):
        sg = self.gradient()
        gradOut = gradIn @ sg
        return gradOut

    def learn(self, gradIn, t):
        N = gradIn.shape[0]

        dJdW = (self.getPrevIn().T @ gradIn)/N
        self.__sW = (Layer.RHO1 * self.__sW) + ((1 - Layer.RHO1) * dJdW)
        self.__rW = (Layer.RHO2 * self.__rW) + ((1 - Layer.RHO2) * (dJdW * dJdW))
        self.__weights -= Layer.ETA * ((self.__sW / (1 - Layer.RHO1 ** t)) / (np.sqrt(self.__rW / (1 - Layer.RHO2 ** t)) + Layer.DELTA))

        dJdb = np.sum(gradIn, axis = 0)/N
        self.__sb = (Layer.RHO1 * self.__sb) + ((1 - Layer.RHO1) * dJdb)
        self.__rb = (Layer.RHO2 * self.__rb) + ((1 - Layer.RHO2) * (dJdb * dJdb))
        self.__biases -= Layer.ETA * ((self.__sb / (1 - Layer.RHO1 ** t)) / (np.sqrt(self.__rb / (1 - Layer.RHO2 ** t)) + Layer.DELTA))

