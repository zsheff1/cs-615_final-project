from .Layer import Layer
from .FullyConnectedLayer import FullyConnectedLayer
from .LinearLayer import LinearLayer
import numpy as np

class ResidualBlock(Layer):
    def __init__(self, *layers):
        super().__init__()

        self.__layers = layers
        dims = [layer.getWeights().shape for layer in layers if hasattr(layer, 'getWeights')]
        sizeIn, sizeOut = dims[0][0], dims[-1][1]

        if sizeIn == sizeOut:
            self.__projection = LinearLayer()
        else:
            self.__projection = FullyConnectedLayer(sizeIn, sizeOut, LinearLayer)
    
    def getLayers(self):
        return self.__layers
        
    def setLayers(self, layers):
        self.__layers = layers
    
    def getProjection(self):
        return self.__projection
    
    def setProjection(self, projection):
        self.__projection = projection

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
        
        print("gradMain shape:", (gradMain).shape)
        print("gradResidual shape:", (gradResidual).shape)
        print("gradMain + gradResidual shape:", (gradMain + gradResidual).shape)

        gradOut = np.stack([gradMain + gradResidual] * len(self.getPrevIn()), axis=0)
        return gradOut

    def backward(self, gradIn):
        sg = self.gradient()

        print("gradIn shape:", gradIn.shape)  # Expected (batch_size, output_dim)
        print("sg shape:", sg.shape)  # Expected (batch_size, output_dim, input_dim)


        gradOut = np.zeros((gradIn.shape[0],sg.shape[2]))

        for n in range(gradIn.shape[0]):
            print("gradIn[n] shape:", np.atleast_2d(gradIn[n]).shape)  # Should be (1, output_dim)
            print("sg[n] shape:", sg[n].shape)  # Should be (output_dim, input_dim)
            gradOut[n] = np.atleast_2d(gradIn[n])@sg[n]
        return gradOut

    def learn(self, gradIn, t):
        grad = gradIn
        for i in range(len(self.__layers) - 1, -1, -1):
            newgrad = self.__layers[i].backward(grad)
            if hasattr(self.__layers[i], 'learn'):
                self.__layers[i].learn(grad, t)
            grad = newgrad

        if hasattr(self.__projection, 'learn'):
            self.__projection.learn(gradIn, t)