import torch
import torch.nn as nn

class logistic_regression(nn.Module):

	def __init__(self, in_size, b = 1):
		super(logistic_regression, self).__init__()
		self.linear = nn.Linear(in_size, b)

	def forward(self, x):
		z = self.linear(x)
		sigma = torch.sigmoid(z)
		return sigma

# Definindo o modelo para 2 dimensões: X = [x1, x2]
custom_2d_model = logistic_regression(2)
print(list(custom_2d_model.parameters()))
print("\n")

# y = sigma(WX + b), W = [w1, w2], 1x2, X = [x1, x2]^T, 2x1
# y = sigma(w1*x1 + w2*x2 + b)
sequential_2d_model = nn.Sequential(nn.Linear(2, 1), nn.Sigmoid())
print(list(sequential_2d_model.parameters()))
print("\n")

# trẽs valores de entrada com duas dimensões
X = torch.tensor([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])

yhat = custom_2d_model(X)
print(yhat)
print("\n")

yhat = sequential_2d_model(X)
print(yhat)
print("\n")

