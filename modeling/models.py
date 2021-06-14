import torch
import torch.nn as nn
import torch.nn.functional as F


class TestModel(nn.Module):
    """This is a simple test model with 3D convolutions and up-convolutions."""
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3))
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3))

        self.upconv1 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(3, 3, 3))
        self.upconv2 = nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=(3, 3, 3))

    def forward(self, x1, x2):
        # Quick check: only work with 3D volumes (extra dimension comes from batch size)
        assert len(x1.shape) == 4 and len(x2.shape) == 4

        # Expand dims (i.e. add channel dim. to beginning)
        x1_, x2_ = x1.unsqueeze(1), x2.unsqueeze(1)

        # Pass to network
        x1_, x2_ = F.relu(self.conv1(x1_)), F.relu(self.conv1(x2_))
        x1_, x2_ = F.relu(self.conv2(x1_)), F.relu(self.conv2(x2_))
        x1_, x2_ = F.relu(self.upconv1(x1_)), F.relu(self.upconv1(x2_))
        x1_, x2_ = self.upconv2(x1_), self.upconv2(x2_)

        # Simply get the differentiable diff. between the two feature maps for now
        y_hat = torch.sigmoid(x2_ - x1_)[:, 0]

        return y_hat
