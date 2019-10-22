#model function
from torch import nn

class NetG(nn.Module):
    def __init__(self,opt):
        super(NetG, self).__init__()
        ngf = opt.ngf  # 生成器feature map数
        self.netg=nn.Sequential(
            nn.ConvTranspose2d(opt.nz, ngf * 8, 4, 1, 0, bias=False),#ngf*8,4,4
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 4, 2, bias=False),#ngf*4,12,12
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),#ngf*2,24,24
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),#ngf,48,48
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),#3,96,96
            nn.Tanh()  # 输出范围 -1~1 故而采用Tanh
            # 输出形状：3 x 96 x 96
        )
    def forward(self, input):
        return self.netg(input)

class NetD(nn.Module):
    def __init__(self, opt):
        super(NetD, self).__init__()
        ndf = opt.ndf
        self.netd = nn.Sequential(
            nn.Conv2d(3, ndf, 4, 2, 1, bias=False),#ngf,48,48
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),#ngf*2,24,24
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),#ngf*4,12,12
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 4, ndf * 8, 4, 4, 2, bias=False),#ngf*8,4,4
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()  # 输出一个数(概率)
        )
    def forward(self, input):
        return self.netd(input).view(-1)