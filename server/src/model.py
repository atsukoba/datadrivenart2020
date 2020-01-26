import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F

device = torch.device("cuda")

ngf = 64
ndf = 64


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, input):
        return input.view(-1, 512, 8, 8)


class FlattenVae(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class UnFlattenVae(nn.Module):
    def forward(self, input, size=1024):
        return input.view(input.size(0), size, 1, 1)


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=1024, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, ),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            FlattenVae()
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            UnFlattenVae(),
            nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
            nn.Sigmoid(),
        )

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).to(device).float()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar


class CVAE(nn.Module):
    def __init__(self, nz, cuda=True):
        super(CVAE, self).__init__()

        self.have_cuda = cuda
        self.nz = nz

        self.encoder = nn.Sequential(
            # input is (nc) x 28 x 28
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 14 x 14
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 7 x 7
            nn.Conv2d(ndf * 2, ndf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1024, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            # nn.Sigmoid()
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(1024, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     nc, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            #             state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            nn.Sigmoid()
            # state size. (nc) x 64 x 64
        )

        self.fc1 = nn.Linear(1024, 512)
        self.fc21 = nn.Linear(512, nz)
        self.fc22 = nn.Linear(512, nz)

        self.fc3 = nn.Linear(nz, 512)
        self.fc4 = nn.Linear(512, 1024)

        self.lrelu = nn.LeakyReLU()
        self.relu = nn.ReLU()
        # self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        conv = self.encoder(x)
        # print("encode conv", conv.size())
        h1 = self.fc1(conv.view(-1, 1024))
        # print("encode h1", h1.size())
        return self.fc21(h1), self.fc22(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        deconv_input = self.fc4(h3)
        # print("deconv_input", deconv_input.size())
        deconv_input = deconv_input.view(-1, 1024, 1, 1)
        # print("deconv_input", deconv_input.size())
        return self.decoder(deconv_input)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.have_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, x):
        # print("x", x.size())
        mu, logvar = self.encode(x)
        # print("mu, logvar", mu.size(), logvar.size())
        z = self.reparametrize(mu, logvar)
        # print("z", z.size())
        decoded = self.decode(z)
        # print("decoded", decoded.size())
        return decoded, mu, logvar


class VGG16VAE(nn.Module):
    """
    VGG16-like Convolutional Encoder-Decoder
    """

    def __init__(self, image_channels=3, h_dim=512*8*8, z_dim=100):
        super(VGG16VAE, self).__init__()

        # 元ネタ
        #         self.encoder = nn.Sequential(
        #             nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
        #             nn.ReLU(),
        #             nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #             nn.ReLU(),
        #             nn.Conv2d(64, 128, kernel_size=4, stride=2),
        #             nn.ReLU(),
        #             nn.Conv2d(128, 256, kernel_size=4, stride=2),
        #             nn.ReLU(),
        #             Flatten()
        #         )
        #         self.decoder = nn.Sequential(
        #             UnFlatten(),
        #             nn.ConvTranspose2d(h_dim, 128, kernel_size=5, stride=2),
        #             nn.ReLU(),
        #             nn.ConvTranspose2d(128, 64, kernel_size=5, stride=2),
        #             nn.ReLU(),
        #             nn.ConvTranspose2d(64, 32, kernel_size=6, stride=2),
        #             nn.ReLU(),
        #             nn.ConvTranspose2d(32, image_channels, kernel_size=6, stride=2),
        #             nn.Sigmoid(),
        #         )

        # Convolutional Encoder
        self.conv2d_1 = nn.Conv2d(3, 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d_2 = nn.Conv2d(64, 64, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d_3 = nn.Conv2d(64, 128, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d_4 = nn.Conv2d(128, 128, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d_5 = nn.Conv2d(128, 256, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d_6 = nn.Conv2d(256, 256, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d_7 = nn.Conv2d(256, 256, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d_8 = nn.Conv2d(256, 512, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))

        self.conv2d_9 = nn.Conv2d(512, 512, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d_10 = nn.Conv2d(512, 512, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d_11 = nn.Conv2d(512, 512, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d_12 = nn.Conv2d(512, 512, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2d_13 = nn.Conv2d(512, 512, kernel_size=(
            3, 3), stride=(1, 1), padding=(1, 1))
        self.maxpool2d_1 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        self.maxpool2d_2 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        self.maxpool2d_3 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        self.maxpool2d_4 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)
        self.maxpool2d_5 = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0, dilation=1, return_indices=True)

        # Full Connection
        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        # Convolutional Decoder
        self.upconv2d_1 = nn.ConvTranspose2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2d_2 = nn.ConvTranspose2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2d_3 = nn.ConvTranspose2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2d_4 = nn.ConvTranspose2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2d_5 = nn.ConvTranspose2d(
            512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.upconv2d_6 = nn.ConvTranspose2d(
            512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2d_7 = nn.ConvTranspose2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2d_8 = nn.ConvTranspose2d(
            256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2d_9 = nn.ConvTranspose2d(
            256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2d_10 = nn.ConvTranspose2d(
            128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2d_11 = nn.ConvTranspose2d(
            128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2d_12 = nn.ConvTranspose2d(
            64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.upconv2d_13 = nn.ConvTranspose2d(
            64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.maxunpool2d_1 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.maxunpool2d_2 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.maxunpool2d_3 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.maxunpool2d_4 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)
        self.maxunpool2d_5 = nn.MaxUnpool2d(kernel_size=2, stride=2, padding=0)

    def encoder(self, input):
        x = self.conv2d_1(input)
        x = nn.ReLU()(x)
        x = self.conv2d_2(x)
        x = nn.ReLU()(x)
        x, self.pooling_idx_1 = self.maxpool2d_1(x)
        x = self.conv2d_3(x)
        x = nn.ReLU()(x)
        x = self.conv2d_4(x)
        x = nn.ReLU()(x)
        x, self.pooling_idx_2 = self.maxpool2d_2(x)
        x = self.conv2d_5(x)
        x = nn.ReLU()(x)
        x = self.conv2d_6(x)
        x = nn.ReLU()(x)
        x = self.conv2d_7(x)
        x = nn.ReLU()(x)
        x, self.pooling_idx_3 = self.maxpool2d_3(x)
        x = self.conv2d_8(x)
        x = nn.ReLU()(x)
        x = self.conv2d_9(x)
        x = nn.ReLU()(x)
        x = self.conv2d_10(x)
        x = nn.ReLU()(x)
        x, self.pooling_idx_4 = self.maxpool2d_4(x)
        x = self.conv2d_11(x)
        x = nn.ReLU()(x)
        x = self.conv2d_12(x)
        x = nn.ReLU()(x)
        x = self.conv2d_13(x)
        x = nn.ReLU()(x)
        x, self.pooling_idx_5 = self.maxpool2d_5(x)
        x = Flatten()(x)
        return x

    def decoder(self, input):
        x = UnFlatten()(input)
        x = self.maxunpool2d_1(x, self.pooling_idx_5)
        x = nn.ReLU()(x)
        x = self.upconv2d_1(x)
        x = nn.ReLU()(x)
        x = self.upconv2d_2(x)
        x = nn.ReLU()(x)
        x = self.upconv2d_3(x)
        x = self.maxunpool2d_2(x, self.pooling_idx_4)
        x = nn.ReLU()(x)
        x = self.upconv2d_4(x)
        x = nn.ReLU()(x)
        x = self.upconv2d_5(x)
        x = nn.ReLU()(x)
        x = self.upconv2d_6(x)
        x = self.maxunpool2d_3(x, self.pooling_idx_3)
        x = nn.ReLU()(x)
        x = self.upconv2d_7(x)
        x = nn.ReLU()(x)
        x = self.upconv2d_8(x)
        x = nn.ReLU()(x)
        x = self.upconv2d_9(x)
        x = self.maxunpool2d_4(x, self.pooling_idx_2)
        x = nn.ReLU()(x)
        x = self.upconv2d_10(x)
        x = nn.ReLU()(x)
        x = self.upconv2d_11(x)
        x = self.maxunpool2d_5(x, self.pooling_idx_1)
        x = nn.ReLU()(x)
        x = self.upconv2d_12(x)
        x = nn.ReLU()(x)
        x = self.upconv2d_13(x)
        x = nn.Sigmoid()(x)
        return x

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size()).cuda().float()
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu = self.fc1(h)
        logvar = self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        z = self.decode(z)
        return z, mu, logvar
