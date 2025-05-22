import numpy as np

class MaxPoolLayer():
    def __init__(self, pool_size, stride):
        self.__prevIn = []
        self.__prevOut = []
        self.pool_size = pool_size
        self.stride = stride

    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn
  
    def setPrevOut(self, out):
        self.__prevOut = out
  
    def getPrevIn(self):
        return self.__prevIn
  
    def getPrevOut(self):
        return self.__prevOut

    @staticmethod
    def maxPool(image, Q, S):
        D, E = image.shape
        H_out = int(np.floor(((D-Q)/S) + 1))
        W_out = int(np.floor(((E-Q)/S) + 1))
        output = np.zeros((H_out, W_out))
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                output[i, j] = np.max(image[i*S:i*S+Q, j*S:j*S+Q])
        return output

    def forward(self, input):
        self.setPrevIn(input)
        output = []
        for _, image in enumerate(input):
            output.append(self.maxPool(image, self.pool_size, self.stride))
        output = np.stack(output, axis=0)
        self.setPrevOut(output)
        return output

    def gradient(self):
        pass

    def backward(self):
        pass
