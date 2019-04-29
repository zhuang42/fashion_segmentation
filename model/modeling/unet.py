import torch
import torch.nn as nn
from torchsummary import summary


class double_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(double_conv, self).__init__()
        padding = kernel_size//2
        self._layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self._layers(x)
        return x


class down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(down, self).__init__()
        self.down = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_channels, out_channels, 3)
        )

    def forward(self, x):
        x = self.down(x)
        return x


class up(nn.Module):
    def __init__(self, in_channels, out_channels):
        #self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        super(up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.dconv = double_conv(in_channels, out_channels, 3)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        x = self.dconv(x)

        return x




class UNet(nn.Module):
    def __init__(self,  num_filters = 32, num_categories=2, num_in_channels=3):
        super(UNet, self).__init__()

        kernel = 3
        padding = 1
        self.inconv = double_conv(num_in_channels, num_filters)
        self.downconv1 = down(num_filters, num_filters*2)
        self.downconv2 = down(num_filters*2, num_filters*4)
        self.downconv3 = down(num_filters * 4, num_filters * 8)
        self.rfconv = double_conv(num_filters*8, num_filters*8)
        self.upconv1 = up(num_filters*(8+4), num_filters*4)
        self.upconv2 = up(num_filters*(4+2), num_filters*2)
        self.upconv3 = up(num_filters *(2+1), num_filters)

        self.finalconv = nn.Conv2d(num_filters, num_categories, 1)

    def forward(self, x):
        out0 = self.inconv(x) #600*400*32
        out1 = self.downconv1(out0) #300 * 200*64
        out2 = self.downconv2(out1) #150 * 100*128
        out3 = self.downconv3(out2) #75 * 50*256
        out4 = self.rfconv(out3) #75 * 50*256
        out5 = self.upconv1(out4, out2) #150*100*384
        out6 = self.upconv2(out5, out1)
        out7 = self.upconv3(out6, out0)
        out_final = self.finalconv(out7)

        return out_final


if __name__ == "__main__":
    model = UNet()
    model.cuda()

    summary(model, (3, 600, 400))

