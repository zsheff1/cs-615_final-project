from .Layer import Layer
from .FullyConnectedLayer import FullyConnectedLayer
from .LinearLayer import LinearLayer
import numpy as np

class ResidualBlock(Layer):
    def __init__(self, *layers):
        super().__init__()

        dims = [layer.getWeights().shape for layer in layers if hasattr(layer, 'getWeights')]
        sizeIn, sizeOut = dims[0][0], dims[-1][1]
        if sizeIn != sizeOut:
            raise ValueError(
                """Input size must match output size.
                Change dimensionality between residual blocks, not within them."""
            )

        self.__layers = layers
        self.__projection = LinearLayer()
    
    def getLayers(self):
        return self.__layers
        
    def setLayers(self, layers):
        self.__layers = layers

    def forward(self, dataIn):
        self.setPrevIn(dataIn)
        projected_residual = self.__projection.forward(dataIn)

        h = dataIn
        for i in range(len(self.__layers)):
            h = self.__layers[i].forward(h)
        
        dataOut = h + projected_residual
        self.setPrevOut(dataOut)
        return dataOut
 
    def gradient(self):
        gradMain = self.__layers[-1].gradient()
        for i in range(len(self.__layers) - 2, -1, -1):
            gradMain = self.__layers[i].backward(gradMain)

        gradResidual = self.__projection.gradient()

        gradOut = np.atleast_2d(gradMain + gradResidual)
        return gradOut

    def backward(self, gradIn):
        sg = self.gradient()

        gradOut = np.zeros((gradIn.shape[0],sg.shape[1]))

        for n in range(gradIn.shape[0]):
            gradOut[n] = np.atleast_2d(gradIn[n])@sg[n]

        return gradOut

    def learn(self, gradIn, t):
        grad = gradIn
        for i in range(len(self.__layers) - 1, -1, -1):
            newgrad = self.__layers[i].backward(grad)
            if hasattr(self.__layers[i], 'learn'):
                self.__layers[i].learn(grad, t)
            grad = newgrad
