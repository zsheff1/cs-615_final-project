from .Layer import Layer
import numpy as np

class SoftmaxLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataAdjusted = dataIn - np.max(dataIn)
        dataExp = np.exp(dataAdjusted)
        sum = np.sum(dataExp)
        y = dataExp / sum
        self.setPrevOut(y)
        return y
    
    def gradient(self):
        return np.stack([np.diag(g) - (g.T @ g) for g in self.getPrevOut()], axis = 0)
