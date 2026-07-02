import numpy as np
import matplotlib.pyplot as plt
import librosa as lb

#############################################################################
#                                                                           #
#    IN ORDER TO GET LIBRARIES TO WORK, FOR WINDOWS RUN FOLLOWING COMMAND:  #
#       pip install numpy matplotlib librosa                                #
#                                                                           #
#############################################################################

class LMSFilter:
    def __init__(self, numTaps, mu):
        self.numTaps = numTaps
        self.mu = mu
        self.weights = np.zeros(numTaps)
        self.buffer = np.zeros(numTaps)

    def reset(self):
        self.weights[:] = 0.0
        self.buffer[:] = 0.0

    #step through filter once
    def update(self, xn, dn):
        #shift over the input buffer
        self.buffer[1:] = self.buffer[:-1]
        self.buffer[0] = xn

        #filter
        yn = np.dot(self.weights, self.buffer)

        #find error signal
        en = dn - yn

        #update the weight

        self.weights = self.weights + self.mu * en * self.buffer

        #return filtered output and error signal
        return yn, en
    
    #run filter over all signals
    def run(self, x, d):
        x = np.asarray(x)
        d = np.asarray(d)
        length = len(x)

        y = np.zeros(length)
        e = np.zeros(length)
        weightHistory = np.zeros((length, self.numTaps))

        for i in range(length):
            y[i], e[i] = self.update(x[i], d[i])
            weightHistory[i] = self.weights

        return{
            "y": y,
            "e": e,
            "weights": self.weights.copy(),
            "weightHistory": weightHistory,
        }


#main func

np.random.seed(0)

source, st = lb.load("sample.mp3", sr=None, mono=True)
numSamples = len(source)
time = np.arange(numSamples)

noise = np.random.randn(numSamples)
noiseInSignal = 0.5 * noise + 0.2 * np.roll(noise, 1)
noisySignal = source + noiseInSignal

filter = LMSFilter(numTaps=100, mu=0.00001)
result = filter.run(noise, noisySignal)

#side note: result["e"] is the error signal, which is the filtered output

#plot
figure, graph = plt.subplots(3, 1, figsize=(10,9))

graph[0].plot(noisySignal, label="Noisy Signal", alpha=0.5)
graph[1].plot(source, label="Source Signal", linewidth=2)
graph[2].plot(result["e"], label="Filtered Signal", linewidth=1)
graph[0].set_title("Adaptive Noise Cancellation")
graph[0].set_xlabel("Sample")
graph[0].legend(loc="upper right")
graph[1].legend(loc="upper right")
graph[2].legend(loc="upper right")

plt.tight_layout()
plt.show()