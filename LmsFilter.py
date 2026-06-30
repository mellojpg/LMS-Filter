import numpy as np
import matplotlib.pyplot as plt

#############################################################################
#                                                                           #
#    IN ORDER TO GET LIBRARIES TO WORK, FOR WINDOWS RUN FOLLOWING COMMAND:  #
#       pip install numpy matplotlib                                        #
#                                                                           #
#############################################################################

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

#noise cancellation part:

fs = 1000
t = np.arange(0, 1, 1/fs)

#source signal
source = np.sin(2 * np.pi * 50 * t)

#noise signal
noise = 0.5 * np.random.randn(len(t))

#desired signal
desired = source + noise

#reference signal
reference = noise + 0.05 * np.random.randn(len(t))


#filter setup
N = 32
mu = 0.01
filter = LMSFilter(N, mu)

#create output/error arrays
output = np.zeros(len(t))
error = np.zeros(len(t))

for n in range(len(t)):
    y, e = filter.update(reference[n], desired[n])
    output[n] = y
    error[n] = e

#update error signal
cleanedSignal = error

#plot the graphs
plt.figure()
plt.plot(t, desired, label="Noisy Signal")
plt.plot(t, cleanedSignal, label="LMS Filtered Signal")
plt.plot(t, source, label="Source Signal")
plt.xlabel("time (s)")
plt.ylabel("amplitude")
plt.legend()
plt.title("LMS Adaptive Noise Cancellation")
plt.show()