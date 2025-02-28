import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Gera um vetor de entradas
z = torch.arange(-100, 100, 0.1).view(-1, 1)
print(z)

# Aplica a função sigmoide
sigmoide = nn.Sigmoid()
yhat = sigmoide(z)
print(yhat)

# Plota os resultados
plt.plot(z.numpy(), yhat.numpy())
plt.show()
