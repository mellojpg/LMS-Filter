import numpy as np
import matplotlib.pyplot as plt

class LMSFilter:
    def __init__(self, numTaps, mu):
        self.numTaps = numTaps
        self.mu = mu
        self.weights = np.zeros(numTaps)
        self.buffer = np.zeros(numTaps)

    def update(self, xn, dn):
        #shift over the input buffer
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = xn

        #filter
        yn = np.dot(self.weights, self.buffer)

        #find error signal
        en = dn - yn

        #update the weight

        self.weights += 2 * self.mu * en * self.buffer

        #return filtered output and error signal
        return yn, en

