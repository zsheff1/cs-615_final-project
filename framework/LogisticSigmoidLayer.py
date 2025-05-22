from .Layer import Layer
import numpy as np

class LogisticSigmoidLayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = 1/(1+np.exp(-dataIn))
        self.setPrevOut(y)
        return y
    
    def gradient(self):
        return np.stack([np.array(g * (1-g)) for g in self.getPrevOut()], axis = 0)

    def backward(self, gradIn):
        sg = self.gradient()
        gradOut = gradIn * sg
        return gradOut
