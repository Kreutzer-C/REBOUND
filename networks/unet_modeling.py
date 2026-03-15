import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.simple_tools import load_config_as_namespace


class FeaturesSegmenter(nn.Module):

    def __init__(self, in_channels=64, out_channels=2):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, out_channels, kernel_size=3, padding=1)

    def forward(self, x_):
        x = F.relu(self.conv1(x_))
        x = F.relu(self.conv2(x))
        out = self.conv3(x)

        return out


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, first_channels=64, only_feature=False, only_logits=True, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.only_feature = only_feature
        self.only_logits = only_logits
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, first_channels)
        self.down1 = Down(first_channels, first_channels * 2)
        self.down2 = Down(first_channels * 2, first_channels * 4)
        self.down3 = Down(first_channels * 4, first_channels * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(first_channels * 8, first_channels * 16 // factor)
        self.up1 = Up(first_channels * 16, first_channels * 8 // factor, bilinear)
        self.up2 = Up(first_channels * 8, first_channels * 4 // factor, bilinear)
        self.up3 = Up(first_channels * 4, first_channels * 2 // factor, bilinear)
        self.up4 = Up(first_channels * 2, first_channels, bilinear)
        if self.only_feature == False:
            self.outc = OutConv(first_channels, n_classes)

    def forward(self, x, only_feature = False):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.only_feature:
            return x
        elif self.only_logits:
            return self.outc(x)
        else:
            return x,self.outc(x)


def build_unet(config_path, img_size=256, num_classes=5):
    config = load_config_as_namespace(config_path)
    return UNet(n_channels=config.in_channels, 
                n_classes=num_classes, 
                first_channels=config.first_channels, 
                only_feature=config.only_feature, 
                only_logits=config.only_logits,
                bilinear=config.bilinear)


if __name__ == "__main__":
    model = UNet(n_channels=1, n_classes=5)
    print(model)
    print(f">>> Parameters: {sum(p.numel() for p in model.parameters()):,}")