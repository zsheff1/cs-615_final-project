import numpy as np

class LogLoss():
    EPSILON = 1e-7

    def eval(self, Y, Yhat):
        Yhat = np.clip(Yhat, LogLoss.EPSILON, 1 - LogLoss.EPSILON)
        J = -(Y * np.log(Yhat) + (1 - Y) * np.log(1 - Yhat))
        return np.mean(J)
    
    def gradient(self, Y, Yhat):
        Yhat = np.clip(Yhat, LogLoss.EPSILON, 1 - LogLoss.EPSILON)
        dJdY = - (Y - Yhat) / (Yhat * (1 - Yhat) + LogLoss.EPSILON)
        return dJdY
