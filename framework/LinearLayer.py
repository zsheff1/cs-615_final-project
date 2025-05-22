from .Layer import Layer
import numpy as np

class LinearLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = dataIn
        self.setPrevOut(y)
        return y
    
    def gradient(self):
        return np.stack([np.ones(g.shape[0]) for g in self.getPrevOut()], axis = 0)

    def backward(self, gradIn):
        sg = self.gradient()
        gradOut = gradIn * sg
        return gradOut
