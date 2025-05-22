import numpy as np

class SquaredError():
    def eval(self, Y, Yhat):
        J = (Y - Yhat) ** 2
        return np.mean(J)
    
    def gradient(self, Y, Yhat):
        return -2 * (Y - Yhat)
