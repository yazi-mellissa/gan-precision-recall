import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    """Fixed by the assignment: MLP 100 -> 256 -> 512 -> 1024 -> 784, LeakyReLU 0.2, tanh output."""

    def __init__(self, g_output_dim):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features * 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features * 2)
        self.fc4 = nn.Linear(self.fc3.out_features, g_output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        return torch.tanh(self.fc4(x))


class Discriminator(nn.Module):
    """Only the generator is fixed, so the discriminator may change.

    Kept:  arch="mlp", sn=True  -> the starter MLP with spectral normalisation on every layer.
    Tried: arch="conv"          -> a small conv net (better FID but much lower recall, see README).
    arch="mlp", sn=False is exactly the starter discriminator (same layers and state-dict keys).
    """

    def __init__(self, d_input_dim, arch="mlp", sn=False):
        super(Discriminator, self).__init__()
        self.arch = arch
        norm = nn.utils.spectral_norm if sn else (lambda m: m)
        if arch == "mlp":
            self.fc1 = norm(nn.Linear(d_input_dim, 1024))
            self.fc2 = norm(nn.Linear(1024, 512))
            self.fc3 = norm(nn.Linear(512, 256))
            self.fc4 = norm(nn.Linear(256, 1))
        else:
            self.conv1 = norm(nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1))    # 28 -> 14
            self.conv2 = norm(nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1))  # 14 -> 7
            self.fc = norm(nn.Linear(128 * 7 * 7, 1))

    def forward(self, x, logits=False):
        """Returns the raw score (logit) if logits=True, else the probability that x is real."""
        if self.arch == "mlp":
            x = F.leaky_relu(self.fc1(x), 0.2)
            x = F.leaky_relu(self.fc2(x), 0.2)
            x = F.leaky_relu(self.fc3(x), 0.2)
            x = self.fc4(x)
        else:
            x = x.view(-1, 1, 28, 28)
            x = F.leaky_relu(self.conv1(x), 0.2)
            x = F.leaky_relu(self.conv2(x), 0.2)
            x = self.fc(x.flatten(1))
        if logits:
            return x
        return torch.sigmoid(x)
