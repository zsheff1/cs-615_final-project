from .Layer import Layer
import numpy as np

class ReLULayer(Layer):
    def __init__(self):
        super().__init__()
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        y = np.where(dataIn <= 0, 0, dataIn)
        self.setPrevOut(y)
        return y
    
    def gradient(self):
        return np.stack([np.array((g == z).astype(int)) for g, z in zip(self.getPrevOut(), self.getPrevIn())], axis = 0)

    def backward(self, gradIn):
        sg = self.gradient()
        gradOut = gradIn * sg
        return gradOut
