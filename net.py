import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    """CNN producing policy logits for 225 actions and a scalar value estimate."""

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=2, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)

        # keep spatial size (15x15) via padding
        self.fc1 = nn.Linear(256 * 15 * 15, 512)
        self.fc2 = nn.Linear(512, 256)
        self.policy_head = nn.Linear(256, 225)
        self.value_head = nn.Linear(256, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        return policy_logits, value.squeeze(-1)

