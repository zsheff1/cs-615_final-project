from .Layer import Layer
import numpy as np

class TanhLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = (np.exp(dataIn)-np.exp(-dataIn))/(np.exp(dataIn)+np.exp(-dataIn))
        self.setPrevOut(y)
        return y
    
    def gradient(self):
        return np.stack([np.array(1 - g**2) for g in self.getPrevOut()], axis = 0)

    def backward(self, gradIn):
        sg = self.gradient()
        gradOut = gradIn * sg
        return gradOut
