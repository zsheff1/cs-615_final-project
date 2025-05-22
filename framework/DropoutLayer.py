from .Layer import Layer
import numpy as np

class DropoutLayer(Layer):
    def __init__(self, probability):
        super().__init__()
        
        self.__probability = probability
        self.__filter = None

    def getProbability(self):
        return self.__probability
    
    def setProbability(self, probability):
        self.__probability = probability
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        self.__filter = (1 / (1 - self.__probability)) * np.random.binomial(1, 1 - self.__probability, size = dataIn.shape)
        dataOut = self.__filter * dataIn
        self.setPrevOut(dataOut)
        return dataOut
    
    def gradient(self):
        return self.__filter
    
    def backward(self, gradIn):
        sg = self.gradient()
        gradOut = gradIn * sg
        return gradOut
