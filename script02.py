import torch
import matplotlib.pyplot as plt

z = torch.arange(-100, 100, 0.1).view(-1, 1)
yhat = torch.sigmoid(z)
plt.plot(z.numpy(), yhat.numpy())
plt.show()
