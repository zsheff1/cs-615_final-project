import numpy as np

class CrossEntropy():
    EPSILON = 1e-7

    def eval(self, Y, Yhat):
        Yhat = np.clip(Yhat, CrossEntropy.EPSILON, 1 - CrossEntropy.EPSILON)
        J = -Y * np.log(Yhat.T)
        return np.mean(J)
    
    def gradient(self, Y, Yhat):
        Yhat = np.clip(Yhat, CrossEntropy.EPSILON, 1 - CrossEntropy.EPSILON)
        return - Y / Yhat
