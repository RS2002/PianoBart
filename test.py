import torch

a=torch.zeros((10,2,3))
b=torch.zeros((10,2,3))
c=torch.cat([a,b],dim=-1)
print(c.shape)