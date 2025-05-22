from .Layer import Layer

from .LinearLayer import LinearLayer
from .ReLULayer import ReLULayer
from .LogisticSigmoidLayer import LogisticSigmoidLayer
from .TanhLayer import TanhLayer
from .SoftmaxLayer import SoftmaxLayer

import numpy as np

class FullyConnectedLayer(Layer):
    # define ADAM constants
    RHO1 = 0.9
    RHO2 = 0.999
    ETA = 0.001
    DELTA = 1e-8

    def __init__(self, sizeIn, sizeOut, activationFunction):
        super().__init__()

        # initialize weights and biases
        # standard initialization
        if activationFunction in [LinearLayer, SoftmaxLayer]:
            range = 1e-4
            self.__weights = np.random.uniform(low = -range, high = range, size = (sizeIn, sizeOut))
            self.__biases = np.random.uniform(low = -range, high = range, size = (1, sizeOut))
        # xavier initialization
        elif activationFunction in [LogisticSigmoidLayer, TanhLayer]:
            range = np.sqrt(6 / (sizeIn + sizeOut))
            self.__weights = np.random.uniform(low = -range, high = range, size = (sizeIn, sizeOut))
            self.__biases = np.zeros((1, sizeOut))
        # he initialization
        elif activationFunction in [ReLULayer]:
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
        self.__sW = (FullyConnectedLayer.RHO1 * self.__sW) + ((1 - FullyConnectedLayer.RHO1) * dJdW)
        self.__rW = (FullyConnectedLayer.RHO2 * self.__rW) + ((1 - FullyConnectedLayer.RHO2) * (dJdW * dJdW))
        self.__weights -= FullyConnectedLayer.ETA * ((self.__sW / (1 - FullyConnectedLayer.RHO1 ** t)) / (np.sqrt(self.__rW / (1 - FullyConnectedLayer.RHO2 ** t)) + FullyConnectedLayer.DELTA))

        dJdb = np.sum(gradIn, axis = 0)/N
        self.__sb = (FullyConnectedLayer.RHO1 * self.__sb) + ((1 - FullyConnectedLayer.RHO1) * dJdb)
        self.__rb = (FullyConnectedLayer.RHO2 * self.__rb) + ((1 - FullyConnectedLayer.RHO2) * (dJdb * dJdb))
        self.__biases -= FullyConnectedLayer.ETA * ((self.__sb / (1 - FullyConnectedLayer.RHO1 ** t)) / (np.sqrt(self.__rb / (1 - FullyConnectedLayer.RHO2 ** t)) + FullyConnectedLayer.DELTA))

