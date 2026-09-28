import torch

x = torch.load("infer/rkd_meta_llavaov_cls_tiny/ImageNet-1K/query/00000000.pt")
y = torch.load("infer/FastVLM-0.5B_base_b16_cls/ImageNet-1K/query/00000000.pt")

print(x['hidden_shape'])

print('===============================')

print(y['hidden_shape'])