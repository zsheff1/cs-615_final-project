from .Layer import Layer
import numpy as np

class InputLayer(Layer):
    def __init__(self, dataIn):
        super().__init__()

        self.__meanX = np.mean(dataIn, axis=0)
        std = np.std(dataIn, axis=0, ddof=1)
        self.__stdX = np.where(std == 0, 1, std)
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        dataZscored = (dataIn - self.__meanX) / self.__stdX
        self.setPrevOut(dataZscored)
        return dataZscored
    
    def gradient(self):
        pass
