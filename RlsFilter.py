import numpy as np

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