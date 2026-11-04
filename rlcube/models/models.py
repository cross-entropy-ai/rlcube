import torch
import torch.nn.functional as F
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        out = self.fc1(F.relu(self.ln1(x)))
        out = self.fc2(F.relu(self.ln2(out)))
        return out + x


class ValueNet(nn.Module):
    """Cost-to-go estimator: predicts how many moves a state is from solved.

    Input: (B, 24, 6) one-hot stickers of a canonical state (see
    rlcube.envs.cube2.canonicalize). Output: (B,) non-negative cost-to-go.
    """

    def __init__(self, hidden_dim=512, num_residual_blocks=4):
        super().__init__()
        self.fc_in = nn.Linear(24 * 6, hidden_dim)
        self.residual_blocks = nn.Sequential(
            *[
                ResidualBlock(hidden_dim, hidden_dim * 2)
                for _ in range(num_residual_blocks)
            ]
        )
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out = F.relu(self.fc_in(x.reshape(x.size(0), -1)))
        out = self.residual_blocks(out)
        return self.fc_out(F.relu(self.ln_out(out))).view(-1)

    def save(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def load(self, filepath: str):
        self.load_state_dict(torch.load(filepath, map_location="cpu"))


if __name__ == "__main__":
    net = ValueNet()
    x = torch.randn(4, 24, 6)
    print("Input shape:", x.shape)
    print("Output shape:", net(x).shape)
    print("Number of parameters:", sum(p.numel() for p in net.parameters()))
