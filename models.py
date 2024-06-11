import numpy as np
import torch
import torch.nn as nn
import os

"""
      score network: input_dim == output_dim 
log-density network: input_dim == n, output_dim == 1

noise conditioned: input_dim == n + 1
"""
class MLPs(nn.Module):
    def __init__(
            self,
            input_dim=2,
            output_dim=1,
            units=[4096, 4096],
            layernorm=False,
            dropout=None,
            last_activation=nn.Identity()
    ):
        super().__init__()
        layers = []
        in_dim = input_dim
        self.layernorm = layernorm

        def block(in_, out_):
            layers = [
                nn.Linear(in_, out_),
                nn.LayerNorm(out_) if self.layernorm else nn.Identity(),
                nn.GELU(),
                nn.Dropout(dropout) if dropout else nn.Identity()
            ]

            return nn.Sequential(*layers)

        for out_dim in units:
            layers.extend([
                block(in_dim, out_dim)
            ])
            in_dim = out_dim

        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(last_activation)
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# --- log-density model ---
class ScoreOrLogDensityNetwork(nn.Module):
    def __init__(self, net, score_network=False):
        """
        For standard MULDE use ScoreOrLogDensityNetwork(MLPs(input_dim=d+1, output_dim=1, units=[4096, 4096]))
        For MSMA/NCSN use ScoreOrLogDensityNetwork(MLPs(input_dim=d+1, output_dim=d, units=[4096, 4096]), use score_network=True)

        Multiscale extension to: https://github.com/Ending2015a/toy_gradlogp/tree/master

        Args:
            net (nn.Module): An log-density function, the output shape of
                the log-density function should be (b, 1). The score is
                computed by grad(-log-density(x))
            score_network (bool, optional): If True, the log-density network is replaced by a score network.
                In this case the grad(-log-density(x)) is not computed, but the output of the network is returned.
                d -> d mapping instead of d -> 1. Defaults to False.
                This is used for the MSMA model.
        """
        super().__init__()
        self.network = net
        self.is_score_network = score_network

    def forward(self, x):
        return self.network(x)

    def score(self, x, return_log_density=False):
        score, log_density = None, None
        if self.is_score_network:
            score = self.network(x)  # log-density network is actually the score network. n_in (+ 1) == n_out
            if return_log_density:  # in order to preserve the coding interface, return zeros for log-densities
                log_density = torch.zeros_like(score[:, 0][:, None])
        else:  # actual MULDE model
            x = x.requires_grad_()
            log_density = self.network(x)
            logp = -log_density.sum()
            score = torch.autograd.grad(logp, x, create_graph=True)[0]  # grad(-log-density(x))

        if return_log_density:
            return score, log_density
        else:
            return score

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        return self
