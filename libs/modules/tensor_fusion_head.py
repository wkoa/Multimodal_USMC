import torch
from torch import nn


class TensorFusionHead(nn.Module):
    def __init__(self):
        super(TensorFusionHead, self).__init__()

    def forward(self, x):
        bs, mn, fd = x.shape  # batch_size, modality_num, feature_dim
        out = [torch.ones((bs, fd+1)) for _ in range(mn)]
        if x.is_cuda:
            out = [n.cuda() for n in out]

        for i in range(mn):
            try:
                out[i] = torch.cat((torch.ones((bs, 1)), x[:, i, :]), dim=1)
            except RuntimeError:
                out[i] = torch.cat((torch.ones((bs, 1)).cuda(), x[:, i, :]), dim=1)
            if i == 1:
                # fusion tensor shape: batch_size, feature_dim+1, feature_dim+1
                fusion_tensor = torch.bmm(out[i-1].unsqueeze(2), out[i].unsqueeze(1))
            elif i > 1:
                fusion_tensor = fusion_tensor.view(bs, (fd+1)**i, 1)
                fusion_tensor = torch.bmm(fusion_tensor, out[i].unsqueeze(1))

        fusion_tensor = fusion_tensor.view(bs, -1)
        return fusion_tensor