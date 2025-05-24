from .Layer import Layer
import numpy as np

class NormalizationLayer(Layer):
    # define constants
    EPSILON = 1e-7

    def __init__(self, sizeIn):
        super().__init__()

        self.__D = sizeIn
        self.__gamma = np.ones((1, sizeIn))
        self.__beta = np.zeros((1, sizeIn))

        # initialize ADAM terms
        self.__sg = np.zeros_like(self.__gamma)
        self.__rg = np.zeros_like(self.__gamma)
        self.__sb = np.zeros_like(self.__beta)
        self.__rb = np.zeros_like(self.__beta)

    def getGamma(self):
        return self.__gamma
    
    def setGamma(self, gamma):
        self.__gamma = gamma
    
    def getBeta(self):
        return self.__beta
    
    def setBeta(self, beta):
        self.__beta = beta
    
    def forward(self, dataIn):
        self.setPrevIn(dataIn)

        mu = np.mean(dataIn, axis = 1, keepdims = True)
        sigma = np.std(dataIn, axis = 1, keepdims = True)

        x_hat = (dataIn - mu) / (sigma + NormalizationLayer.EPSILON)

        z = self.__gamma * x_hat + self.__beta

        self.setPrevOut(z)
        return z

    def gradient(self):
        x = self.getPrevIn()

        mu = np.mean(x, axis=1, keepdims=True)
        sigma2 = np.var(x, axis=1, keepdims=True)
        sigma = np.sqrt(sigma2 + NormalizationLayer.EPSILON)

        return (self.__gamma / sigma)[:, :, np.newaxis] * (
            np.identity(self.__D)[np.newaxis, :, :]
            - (1 / self.__D)
            - (1 / (self.__D * sigma2 + NormalizationLayer.EPSILON))[:, :, np.newaxis]
            * np.einsum('nd,ne->nde', x - mu, x - mu)
        )

    def learn(self, gradIn, t):
        dJdbeta = np.mean(gradIn, axis = 0, keepdims = True)
        self.__sb = (Layer.RHO1 * self.__sb) + ((1 - Layer.RHO1) * dJdbeta)
        self.__rb = (Layer.RHO2 * self.__rb) + ((1 - Layer.RHO2) * (dJdbeta * dJdbeta))
        self.__beta -= Layer.ETA * ((self.__sb / (1 - Layer.RHO1 ** t)) / (np.sqrt(self.__rb / (1 - Layer.RHO2 ** t)) + Layer.DELTA))

        dJdgamma = np.mean(gradIn * self.getPrevOut(), axis = 0, keepdims = True)
        self.__sg = (Layer.RHO1 * self.__sg) + ((1 - Layer.RHO1) * dJdgamma)
        self.__rg = (Layer.RHO2 * self.__rg) + ((1 - Layer.RHO2) * (dJdgamma * dJdgamma))
        self.__gamma -= Layer.ETA * ((self.__sg / (1 - Layer.RHO1 ** t)) / (np.sqrt(self.__rg / (1 - Layer.RHO2 ** t)) + Layer.DELTA))

    