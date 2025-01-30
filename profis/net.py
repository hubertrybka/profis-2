import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MolecularVAE(nn.Module):
    def __init__(self, latent_size=32, alphabet_size=30, dropout=0, eps_coef=1):
        super(MolecularVAE, self).__init__()

        self.conv_1 = nn.Conv1d(100, 9, kernel_size=9)
        self.conv_2 = nn.Conv1d(9, 9, kernel_size=9)
        self.conv_3 = nn.Conv1d(9, 10, kernel_size=11)
        self.fc1 = nn.Linear(40, 435)
        self.fc2 = nn.Linear(435, latent_size)
        self.fc3 = nn.Linear(435, latent_size)

        self.fc4 = nn.Linear(latent_size, 256)
        self.gru = nn.GRU(256, 512, 3, batch_first=True, dropout=dropout)
        self.fc5 = nn.Linear(512, alphabet_size)

        self.eps_coef = eps_coef
        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=1)

    def encode(self, x):
        x = self.relu(self.conv_1(x))
        x = self.relu(self.conv_2(x))
        x = self.relu(self.conv_3(x))
        x = self.selu(self.fc1(x.view(x.size(0), -1)))
        mu = self.fc2(x)
        logvar = self.fc3(x)
        return mu, logvar

    def sampling(self, z_mean, z_logvar):
        epsilon = self.eps_coef * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = self.selu(self.fc4(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 100, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = self.softmax(self.fc5(out_reshape))
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar


class VaeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_hat, y, z_mean, z_logvar):
        xent_loss = F.binary_cross_entropy(x_hat, y, reduction="sum")
        kl_loss = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp())
        return xent_loss, kl_loss


class Profis(nn.Module):
    def __init__(
        self,
        fp_size=2048,
        fc1_size=1024,
        fc2_size=1024,
        latent_size=32,
        hidden_size=512,
        gru_layers=3,
        eps_coef=1,
        dropout=0,
        alphabet_size=30,
    ):
        super(Profis, self).__init__()

        self.fp_size = fp_size
        self.fc1 = nn.Linear(fp_size, fc1_size)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc31 = nn.Linear(fc2_size, latent_size)
        self.fc32 = nn.Linear(fc2_size, latent_size)
        self.fc4 = nn.Linear(latent_size, 256)
        self.gru = nn.GRU(
            256, hidden_size, gru_layers, batch_first=True, dropout=dropout
        )
        self.fc5 = nn.Linear(hidden_size, alphabet_size)

        self.relu = nn.ReLU()
        self.selu = nn.SELU()
        self.softmax = nn.Softmax(dim=1)
        self.eps_coef = eps_coef

    def encode(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        mu = self.fc31(h2)
        logvar = self.fc32(h2)
        return mu, logvar

    def sampling(self, z_mean, z_logvar):
        epsilon = self.eps_coef * torch.randn_like(z_logvar)
        return torch.exp(0.5 * z_logvar) * epsilon + z_mean

    def decode(self, z):
        z = self.selu(self.fc4(z))
        z = z.view(z.size(0), 1, z.size(-1)).repeat(1, 100, 1)
        output, hn = self.gru(z)
        out_reshape = output.contiguous().view(-1, output.size(-1))
        y0 = self.softmax(self.fc5(out_reshape))
        y = y0.contiguous().view(output.size(0), -1, y0.size(-1))
        return y

    def forward(self, x):
        z_mean, z_logvar = self.encode(x)
        z = self.sampling(z_mean, z_logvar)
        return self.decode(z), z_mean, z_logvar


class Annealer:
    """
    This class is used to anneal the KL divergence loss over the course of training VAEs.
    After each call, the step() function should be called to update the current epoch.
    """

    def __init__(self, total_steps, shape, baseline=0.0, cyclical=False, disable=False):
        """
        Parameters:
            total_steps (int): Number of epochs to reach full KL divergence weight.
            shape (str): Shape of the annealing function. Can be 'linear', 'cosine', or 'logistic'.
            baseline (float): Starting value for the annealing function [0-1]. Default is 0.0.
            cyclical (bool): Whether to repeat the annealing cycle after total_steps is reached.
            disable (bool): If true, the __call__ method returns unchanged input (no annealing).
        """
        self.total_steps = total_steps
        self.current_step = 0
        self.cyclical = cyclical
        self.shape = shape
        self.baseline = baseline
        if disable:
            self.shape = "none"
            self.baseline = 0.0

    def __call__(self, kld):
        """
        Args:
            kld (torch.tensor): KL divergence loss
        Returns:
            out (torch.tensor): KL divergence loss multiplied by the slope of the annealing function.
        """
        out = kld * self.slope()
        return out

    def slope(self):
        if self.shape == "linear":
            y = self.current_step / self.total_steps
        elif self.shape == "cosine":
            y = (math.cos(math.pi * (self.current_step / self.total_steps - 1)) + 1) / 2
        elif self.shape == "logistic":
            exponent = (self.total_steps / 2) - self.current_step
            y = 1 / (1 + math.exp(exponent))
        elif self.shape == "none":
            y = 1.0
        else:
            raise ValueError(
                "Invalid shape for annealing function. Must be linear, cosine, or logistic."
            )
        y = self.add_baseline(y)
        return y

    def step(self):
        if self.current_step < self.total_steps:
            self.current_step += 1
        if self.cyclical and self.current_step >= self.total_steps:
            self.current_step = 0
        return

    def add_baseline(self, y):
        y_out = y * (1 - self.baseline) + self.baseline
        return y_out

    def cyclical_setter(self, value):
        if value is not bool:
            raise ValueError(
                "Cyclical_setter method requires boolean argument (True/False)"
            )
        else:
            self.cyclical = value
        return
