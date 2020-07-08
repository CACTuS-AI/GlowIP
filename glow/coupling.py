import torch
import torch.nn as nn
import numpy as np
from .net import NN

# device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class CouplingLayer(nn.Module):
    def __init__(self, channels, coupling, coupling_bias, device, nn_init_last_zeros=False):
        super(CouplingLayer, self).__init__()
        self.coupling = coupling
        self.channels = channels
        self.coupling_bias = coupling_bias
        if self.coupling == "affine":
            self.net = NN(channels_in=self.channels // 2, channels_out=self.channels,
                          device=device, init_last_zeros=nn_init_last_zeros)
        elif self.coupling == "additive":
            self.net = NN(channels_in=self.channels // 2, channels_out=self.channels // 2,
                          device=device, init_last_zeros=nn_init_last_zeros)
        else:
            raise "only affine and additive coupling is implemented"
        self.to(device)

    def forward(self, x, logdet=None, reverse=False):
        n, c, h, w = x.size()

        if not reverse:
            # affine coupling layer
            if self.coupling == "affine":
                xa, xb = self.split(x, mode="split-by-chunk")
                s_and_t = self.net(xb)
                s, t = self.split(s_and_t, mode="split-by-alternating")
                s = torch.sigmoid(s + 2.) + self.coupling_bias
                #                t       = torch.tanh(t)
                ya = s * xa + t
                #                ya      = torch.exp(torch.log(s+1e-6)) * xa + t
                y = torch.cat([ya, xb], dim=1)
                logdet = logdet + torch.log(s).view(n, -1).sum(-1)
                assert not np.isnan(
                    y.mean().item()), "nan in coupling forward: s=%0.4f, x=%0.4f, xa=%0.4f, ya=%0.4f, t=%0.3f" % (
                    s.mean().item(), x.mean().item(), xa.mean().item(), ya.mean().item(), t.mean().item())
                assert not np.isinf(
                    y.mean().item()), "inf in coupling forward: s=%0.4f, x=%0.4f, xa=%0.4f, ya=%0.4f, t=%0.3f" % (
                    s.mean().item(), x.mean().item(), xa.mean().item(), ya.mean().item(), t.mean().item())
                return y, logdet
            # additive coupling layer
            if self.coupling == "additive":
                xa, xb = self.split(x, mode="split-by-chunk")
                t = self.net(xb)
                ya = xa + t
                y = torch.cat([ya, xb], dim=1)
                assert not np.isnan(
                    y.mean().item()), "nan in coupling forward: s=%0.4f, x=%0.4f, xa=%0.4f, ya=%0.4f, t=%0.3f" % (
                    s.mean().item(), x.mean().item(), xa.mean().item(), ya.mean().item(), t.mean().item())
                assert not np.isinf(
                    y.mean().item()), "inf in coupling forward: s=%0.4f, x=%0.4f, xa=%0.4f, ya=%0.4f, t=%0.3f" % (
                    s.mean().item(), x.mean().item(), xa.mean().item(), ya.mean().item(), t.mean().item())
                return y, logdet

        if reverse:
            # affine coupling layer
            if self.coupling == "affine":
                xa, xb = self.split(x, mode="split-by-chunk")
                s_and_t = self.net(xb)
                s, t = self.split(s_and_t, mode="split-by-alternating")
                s = torch.sigmoid(s + 2.) + self.coupling_bias
                #                t       = torch.tanh(t)
                ya = (xa - t) / s
                #                ya      = (xa  - t) * torch.exp(-torch.log(s+1e-6))
                y = torch.cat([ya, xb], dim=1)
                assert not np.isnan(
                    y.mean().item()), "nan in coupling reverse: s=%0.4f, x=%0.4f, xa=%0.4f, ya=%0.4f, t=%0.3f" % (
                    s.mean().item(), x.mean().item(), xa.mean().item(), ya.mean().item(), t.mean().item())
                assert not np.isinf(
                    y.mean().item()), "inf in coupling reverse: s=%0.4f, x=%0.4f, xa=%0.4f, ya=%0.4f, t=%0.3f" % (
                    s.mean().item(), x.mean().item(), xa.mean().item(), ya.mean().item(), t.mean().item())
                return y
            # additive coupling layer
            if self.coupling == "additive":
                xa, xb = self.split(x, mode="split-by-chunk")
                t = self.net(xb)
                ya = (xa - t)
                y = torch.cat([ya, xb], dim=1)
                return y

    def split(self, x, mode):
        if mode == "split-by-chunk":
            xa = x[:, :self.channels // 2, :, :]
            xb = x[:, self.channels // 2:, :, :]
            return xa, xb
        if mode == "split-by-alternating":
            xa = x[:, 0::2, :, :].contiguous()
            xb = x[:, 1::2, :, :].contiguous()
            return xa, xb


if __name__ == "__main__":
    size = (16, 64, 32, 32)
    coupling = CouplingLayer(channels=64, coupling="affine", device=device, nn_init_last_zeros=False)
    x = torch.tensor(np.random.normal(5, 10, size), dtype=torch.float, device=device)
    logdet = torch.tensor(0, dtype=torch.float, device=device, requires_grad=True)
    y, logdet = coupling(x, logdet=logdet, reverse=False)
    x_rev = coupling(y, reverse=True)
    loss_rev = torch.norm(x_rev - x)
    print(loss_rev.item())