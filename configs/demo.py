import torch
import numpy as np
#
# dist = torch.randn(1, 4, 4)
# clo_label = torch.tensor([1,1,3,4])
# row_labels = torch.tensor([1,1,3,4])
# matches = (row_labels.unsqueeze(1) ==
#            clo_label.unsqueeze(0)).bool()  # [n_r, n_c]
# diffenc = torch.logical_not(matches)  # [n_r, n_c]
# p, n, _ = dist.size()
# # print(matches)
# # print("&&&&&&&&&&&&&",torch.sum(matches))
# # print("*********matches",matches.size(),diffenc.size())
# # print("&&&&&&&&&&&&&&",dist.size(),dist[:, matches].size())
# # &&&&&&&&&&&&&& torch.Size([1, 256, 256]) torch.Size([1, 3328])
# # dist_flattened = dist.view(-1)
# # ap_dist = dist_flattened.masked_select(matches.view(-1)).view(p, n, -1, 1)
# ap_dist = dist[:, matches].view(p, n, -1, 1)
# # an_dist = dist_flattened.masked_select(diffenc.view(-1)).view(p, n, 1, -1)
# an_dist = dist[:, diffenc].view(p, n, 1, -1)

# 随机生成一个1x4x3的张量
X = torch.randn(1, 2, 6, 5)
print(X)
# y = torch.sum(X[:,[1, 2],:],dim=1)/2.0
avg_node= torch.mean(X[:, :, [0, 3], :], dim=2)
# 把X
print(avg_node)

Y = torch.zeros(X.size(0), X.size(1), X.size(2) - 1, X.size(3))
Y[:, :, [1, 2, 3,4], :] = X[:, :, [4, 1, 5, 2], :]
Y[:, :, 0, :] = avg_node
print("**",Y)
