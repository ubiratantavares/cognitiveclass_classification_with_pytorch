import torch
import torch.nn as nn

# Modelo linear seguido de uma função sigmoide
# A equação do classificador linear: z = wx + b = 1x + 1
# A função sigmoide é aplicada ao resultado da equação linear
# A função sigmoide é definida como: sigma(z) = 1 / (1 + exp(-z))
model = nn.Sequential(nn.Linear(1, 1), nn.Sigmoid())
x = torch.tensor([[10.0]])
yhat = model(x)
print(yhat)
