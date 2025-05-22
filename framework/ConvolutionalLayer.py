import numpy as np

class ConvolutionalLayer():
    def __init__(self, kernel_size):
        self.__prevIn = []
        self.__prevOut = []
        self.__kernel = np.random.uniform(low = -1e-4, high = 1e-4, size = (kernel_size, kernel_size))

    def setPrevIn(self, dataIn):
        self.__prevIn = dataIn
  
    def setPrevOut(self, out):
        self.__prevOut = out
    
    def setKernels(self, kernel):
        self.__kernel = kernel
  
    def getPrevIn(self):
        return self.__prevIn
  
    def getPrevOut(self):
        return self.__prevOut
    
    def getKernels(self):
        return self.__kernel
    
    @staticmethod
    def crossCorrelate2D(kernel, image):
        H, W = image.shape
        M = kernel.shape[0]
        output = np.zeros((H - M + 1, W - M + 1))
        for i in range(H - M + 1):
            for j in range(W - M + 1):
                output[i,j] = np.sum(image[i:i+M, j:j+M] * kernel)
        return output

    def forward(self, input):
        self.setPrevIn(input)
        kernel = self.getKernels()
        output = []
        for _, image in enumerate(input):
            output.append(self.crossCorrelate2D(kernel, image))
        output = np.stack(output, axis=0)
        self.setPrevOut(output)
        return output

    def gradient(self):
        pass

    def backward(self):
        pass

    def updateKernels(self):
        pass
