import torch.nn.functional as F
from .unet_parts import *
from .aspp import *

class Up2(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, kernel_size=3, group_size=1):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1)
        self.conv = DoubleConv(out_channels * 2, out_channels, kernel_size, group_size)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.conv1(x1)
        # input is CHW
        diffY = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffX = torch.tensor([x2.size()[3] - x1.size()[3]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class SpoonNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(SpoonNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc1 = DoubleConv(n_channels, 64, 1)
        # self.inc1 = DoubleConv(32, 64, 1)
        self.inc2 = DoubleConv(64, 128, 1)
        self.inc3 = DoubleConv(128, 128, 1)
        self.inc3 = DoubleConv(128, 64, 1)
        self.inc4 = DoubleConv(64, 3, 1)
        self.down1 = Down(3, 3*32, 3, 3)
        self.down2 = Down(3*32, 3*64, 3, 3)
        self.up1 = Up2(3*64, 3*32, 3, 3)
        self.up2 = Up2(3*32, 3, 1)
        # self.Att = Attention_block_groups(F_g=3*32,F_l=3)
        self.outc1 = OutConv(3, n_classes)
        self.outc2 = OutConv(3, n_classes)

    def forward(self, x):
        x1 = self.inc4(self.inc3(self.inc2(self.inc1(x))))
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.up1(x3, x2)
        x = self.up2(x, x1)
        # x = self.Att(x, x1)
        logits = self.outc1(x)
        return [logits, self.outc2(x1)]
