import numpy as np
import matplotlib.pyplot as plt
import librosa

class RLS:
	def __init__(self, n_features, forgetting= 0.99, delta=1000):
		self.lambda_ = forgetting
		self.w = np.zeroes((n_features, 1))
		self.P = delta * np.eye(n_features)
	
	def update(self, x, d):
		x = x.reshape(-1, 1)

		# Gain vector
		Px = self.P @ x
		g = Px / (self.lambda_ + x.T @ Px)

		# Prediction
		y = float(self.w.T @ x)
		
		# Error
		e = d - y
		
		# Update Weights
		self.w += g * e
		
		# Update Covariance
		self.P = (self.P - g @ x.T @ self.P) / self.lambda_

		return y, e

fs = 1000 
t = np.linspace(0, 1, fs)

clean = np.sin(2 * np.pi * 5 * t)

noise = 0.3 * np.random.randn(len(t))

noisy = clean + noise



#plot the graphs

# Clean signal
plt.figure(figsize=(8,4))
plt.plot(t, clean)
plt.title("Clean Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Noise
plt.figure(figsize=(8,4))
plt.plot(t, noise)
plt.title("Noise Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

# Noisy signal
plt.figure(figsize=(8,4))
plt.plot(t, noisy)
plt.title("Noisy Signal")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(True)

plt.show()