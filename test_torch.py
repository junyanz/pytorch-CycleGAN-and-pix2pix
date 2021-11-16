import torch
print(torch.__version__)
torch.cuda.init()
print(torch.randn(1, device='cuda'))