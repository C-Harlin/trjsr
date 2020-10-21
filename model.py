import math
import torch
from torch import nn
from torchsummary import summary

class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
        block8 = [UpsampleBLock(64, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(64, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        block5 = self.block5(block4)
        block6 = self.block6(block5)
        block7 = self.block7(block6)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2
class MyGenerator(nn.Module):
    def __init__(self, scale_factor):
        kernels = 16
        upsample_block_num = int(math.log(scale_factor, 2))

        super(MyGenerator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, kernels, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(kernels)
        self.block3 = ResidualBlock(kernels)
        self.block4 = ResidualBlock(kernels)
        self.block5 = ResidualBlock(kernels)
        self.block6 = ResidualBlock(kernels)
        self.block7 = nn.Sequential(
            nn.Conv2d(kernels, kernels, kernel_size=3, padding=1),
            nn.BatchNorm2d(kernels)
        )
        block8 = [UpsampleBLock(kernels, 2) for _ in range(upsample_block_num)]
        block8.append(nn.Conv2d(kernels, 1, kernel_size=9, padding=4))
        self.block8 = nn.Sequential(*block8)

    def forward(self, x):
        block1 = self.block1(x)
        block2 = self.block2(block1)
        block3 = self.block3(block2)
        block4 = self.block4(block3)
        # block5 = self.block5(block4)
        # block6 = self.block6(block5)
        block7 = self.block7(block4)
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )

    def forward(self, x):
        # batch_size = x.size(0)
        # return torch.sigmoid(self.net(x).view(batch_size))
        return self.net(x)

class MyDiscriminator(nn.Module):
    def __init__(self):
        super(MyDiscriminator, self).__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2, 2),
            nn.Conv2d(1, 16, kernel_size=3, stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8, 1, kernel_size=3, padding=1),
            )
        # self.net = nn.Sequential(
        #     nn.Conv2d(1, 16, kernel_size=3, padding=1),
        #     nn.LeakyReLU(0.2),
        #
        #     nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.2),
        #
        #     nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.2),
        #
        #     nn.Conv2d(16, 16, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(16),
        #     nn.LeakyReLU(0.2),
        #
        #     nn.Conv2d(16, 1, kernel_size=3, padding=1),
        # )
        # self.fc1 = nn.Linear(32*40,1000)
        # self.fc2 = nn.Linear(1024,512)
        # self.fc3 = nn.Linear(2048,1024)
    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 32*40)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        # identity_data = x
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        # output = torch.add(output, identity_data)
        return x + residual


class UpsampleBLock(nn.Module):
    def __init__(self, in_channels, up_scale):
        super(UpsampleBLock, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * up_scale ** 2, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(up_scale)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x

class lowdim(nn.Module):
    def __init__(self):
        super(lowdim, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            # nn.Sigmoid()
            # nn.MaxPool2d(2, 2),
        )

    def forward(self, x):
        x = self.net(x)
        x = x.view(-1, 32*41)
        # x = self.fc1(x)
        # x2 = self.fc2(x1)
        # x = self.fc3(x)
        return x

# net = MyGenerator(2)
# net = MyDiscriminator()
# net = Discriminator()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = net.to(device)
# checkpoint = torch.load('checkpoint/bestmodel_G_2.pt')
# start_epoch = checkpoint["epoch"]
# net.load_state_dict(checkpoint["netG"])
# summary(net, (1, 128, 162))
# summary(net, (1, 256, 324))
# class model(nn.Module):
#     def __init__(self):
#         super(model, self).__init__()
#         a = list(net.children())
#         block = [0,1,2,6]
#         c = [a[i] for i in block]
#         self.feature = nn.Sequential(*c)
#         # for param in self.feature.parameters():
#         #     param.requires_grad = False
#     def forward(self, x):
#         out = self.feature(x)
#         return out
# netContent = model()
# summary(netContent, (1, 128, 162))
# # print('# generator parameters:', sum(param.numel() for param in net.parameters()))
# netD = lowdim()
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# netD = netD.to(device)
# summary(netD, (64, 128, 162))
# netG = Generator(2)
# netG.to(device)
# summary(netG, (1, 256, 322))