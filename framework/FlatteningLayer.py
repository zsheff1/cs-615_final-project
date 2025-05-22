import numpy as np

class FlatteningLayer():
    def __init__(self):
        self.__prevIn = []
        self.__prevOut = []

    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn
  
    def setPrevOut(self, out):
        self.__prevOut = out
  
    def getPrevIn(self):
        return self.__prevIn
  
    def getPrevOut(self):
        return self.__prevOut
    
    def forward(self, input):
        self.setPrevIn(input)
        output = input.reshape(input.shape[0], 1, -1, order='F')
        self.setPrevOut(output)
        return output
