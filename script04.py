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


custom_model = logistic_regression(1)
print(list(custom_model.parameters()))
print("\n")

sequential_model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
print(list(sequential_model.parameters()))
print("\n")

x = torch.tensor([[10.0]])

yhat = custom_model(x)
print(yhat)
print("\n")

yhat = sequential_model(x)
print(yhat)
print("\n")

x = torch.tensor([[1.0], [100]])

yhat = custom_model(x)
print(yhat)
print("\n")

yhat = sequential_model(x)
print(yhat)

