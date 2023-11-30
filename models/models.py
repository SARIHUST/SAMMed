import torch
import torch.nn as nn
import torch.nn.functional as F

class CorruptionEncoder(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.dropout1 = nn.Dropout2d(0.3)
        self.conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.dropout2 = nn.Dropout2d(0.5)
        self.conv3 = nn.Conv2d(512, 256, 3, 1, 1)
        self.dropout3 = nn.Dropout2d(0.3)
        self.conv4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.dropout4 = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.dropout1(F.relu(self.conv1(x)))
        x = self.dropout2(F.relu(self.conv2(x)))
        x = self.dropout3(F.relu(self.conv3(x)))
        x = self.dropout4(F.relu(self.conv4(x)))
        return x

class ImageDecoder(nn.Module):
    def __init__(self, latent_dim):
        super(ImageDecoder, self).__init__()
        self.latent_dim = latent_dim

        # 反卷积层
        self.deconv1 = nn.ConvTranspose2d(latent_dim, 256, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv5 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.clone().requires_grad_(True)
        x = self.deconv1(x)

        x = self.relu(x)
        x = self.deconv2(x)

        x = self.relu(x)
        x = self.deconv3(x)

        x = self.relu(x)
        x = self.deconv4(x)

        x = self.relu(x)
        x = self.deconv5(x)

        x = torch.tanh(x)
        return x