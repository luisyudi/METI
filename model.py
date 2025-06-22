import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Conv2d(1, 16, 3, stride=2, padding=1)  # 28->14
        self.enc2 = nn.Conv2d(16, 32, 3, stride=2, padding=1) # 14->7
        self.enc3 = nn.Conv2d(32, 64, 7)                      # 7->1
        # Decoder
        self.dec1 = nn.ConvTranspose2d(64, 32, 7)             # 1->7
        self.dec2 = nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1) # 7->14
        self.dec3 = nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1)  # 14->28

    def encode(self, x):
        x = F.relu(self.enc1(x))
        x = F.relu(self.enc2(x))
        x = F.relu(self.enc3(x))
        return x

    def decode(self, z):
        z = F.relu(self.dec1(z))
        z = F.relu(self.dec2(z))
        z = torch.sigmoid(self.dec3(z))
        return z

    def forward(self, x):
        z = self.encode(x)
        out = self.decode(z)
        return out