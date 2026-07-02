import numpy as np
import matplotlib.pyplot as plt
import librosa

class RLS:
	###
	# RLS Filter
	# learns filter coefficients by minimizing the aquared predictions
	###
	def __init__(self, n_features, forgetting= 0.99, delta=1000):
		#forgetting factor
		self.lambda_ = forgetting
		#filter weights initalized as zero
		self.w = np.zeros((n_features, 1))
		#initalizes inverse covariance matrix
		self.P = delta * np.eye(n_features)
	
	def update(self, x, d):
		# converts input vecotr into col vector
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

###
# Generates a clean sine wave and adds noise
###
fs = 1000 #frequency
t = np.linspace(0, 1, fs)

clean = np.sin(2 * np.pi * 5 * t)


noise = 0.3 * np.random.randn(len(t))

noisy = clean + noise



# Plot the graphs
fig, axs = plt.subplots(3, 1, figsize=(10,8))

# Clean signal
axs[0].plot(t, clean)
axs[0].set_title("Clean Signal")
axs[0].set_ylabel("Amplitude")
axs[0].grid(True)

# Noise
axs[1].plot(t, noise)
axs[1].set_title("Noise Signal")
axs[1].set_ylabel("Amplitude")
axs[1].grid(True)

# Noisy signal
axs[2].plot(t, noisy)
axs[2].set_title("Noise Signal")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Amplitude")
axs[2].grid(True)

plt.tight_layout()
plt.show()