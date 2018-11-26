import torch
import torch.nn as nn
import torch.nn.functional as F


class Autoencoder(nn.Module):

    # assumes that an image is square
    # kernel_size should be odd
    def __init__(self, img_size, channels, kernel_size, dropout, final_activation=nn.Tanh()):
        super().__init__()
        # TODO: use nn.ReflectionPad2d to allow even numbered kernel sizes
        # Padding to ensure that input and output dims are the same
        pad = (int)((kernel_size - 1)/2)
        self.kernel_size = kernel_size
        self.img_size = img_size
        self.channels = channels
        self.pad = pad

        fc_nodes = (int)(8*8*channels)

        # Output of encoder is a 
        self.encoder = nn.Sequential(
            nn.Conv2d(
                3,              # input channels
                channels,       # output channels
                kernel_size,    # kernel size
                stride=1,
                padding=pad,
            ),
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=pad),
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size, stride=2, padding=pad),
            nn.ELU(),

            nn.Dropout2d(dropout),

            # Image size halved

            nn.Conv2d(channels, channels*2, kernel_size, stride=1, padding=pad),
            nn.ELU(),
            nn.Conv2d(channels*2, channels*2, kernel_size, stride=2, padding=pad),
            nn.ELU(),

            # Image size quartered

            nn.Conv2d(channels*2, channels*3, kernel_size, stride=1, padding=pad),
            nn.ELU(),
            nn.Conv2d(channels*3, channels*3, kernel_size, stride=2, padding=pad),
            nn.ELU(),


            # Image size one eigth

            nn.Conv2d(channels*3, channels*4, kernel_size, stride=1, padding=pad),
            nn.ELU(),
            nn.Conv2d(channels*4, channels*4, kernel_size, stride=2, padding=pad),
            nn.ELU(),

            # Image size on sixteenth

            nn.Conv2d(channels*4, channels*5, kernel_size, stride=1, padding=pad),
            nn.ELU(),
            nn.Conv2d(channels*5, channels*5, kernel_size, stride=1, padding=pad),
            nn.ELU(),

            nn.Dropout2d(dropout),
            # nn.Conv2d(channels, channels, kernel_size, stride=2, padding=pad),
            # nn.ELU(),

        )

        self.fc1 = nn.Linear(8*8*channels*5, 64)
        self.fc2 = nn.Linear(64, 8*8*channels)

        self.decoder = nn.Sequential(
            # Image size one sixteenth
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=pad),
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=pad),
            nn.ELU(),

            nn.Dropout2d(dropout),

            nn.UpsamplingNearest2d(scale_factor=2),
           
            # Image size one eigth
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=pad),
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=pad),
            nn.ELU(),


            nn.UpsamplingNearest2d(scale_factor=2),

            # Image size one quarter
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=pad),
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=pad),
            nn.ELU(),

            nn.UpsamplingNearest2d(scale_factor=2),

            # Image size halved
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=pad),
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=pad),
            nn.ELU(),

            nn.UpsamplingNearest2d(scale_factor=2),

            # Image size normal
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=pad),
            nn.ELU(),
            nn.Conv2d(channels, channels, kernel_size, stride=1, padding=pad),
            nn.ELU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(channels, 3, kernel_size, stride=1, padding=pad),
            final_activation,
        )

    def forward(self, x):
        num_examples = x.size()[0]
        x = self.encoder(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = x.view(num_examples, self.channels, 8, 8)
        # x = x.view(num_examples, self.channels, 16, 16)
        x = self.decoder(x)
        return x


class BEGAN():
    def __init__(
        self, 
        img_size, 
        channels, 
        kernel_size, 
        gamma, 
        lmbda, 
        d_dropout, 
        g_dropout,
    ):
        self.device = torch.device('cuda:0')
        self.img_size = img_size
        self.discriminator = Autoencoder(
            img_size, 
            channels, 
            kernel_size, 
            d_dropout, 
            nn.ELU()
        ).to(self.device)
        self.generator = Autoencoder(
            img_size, 
            channels, 
            kernel_size, 
            g_dropout, 
            nn.Tanh()
        ).to(self.device)
        self.k = torch.tensor(0.0).to(self.device)
        self.lmbda = torch.tensor(lmbda).to(self.device)
        self.gamma = torch.tensor(gamma).to(self.device)
        self.discrim_optimizer = None
        self.gen_optimizer = None
        self.batch_conv_measure = torch.tensor(0.0).to(self.device)
        self.batch_d_reconstr_loss = torch.tensor(0.0).to(self.device)
        self.batch_g_reconstr_loss = torch.tensor(0.0).to(self.device)


    def discriminator_loss(self, d_reconstr_loss, g_reconstr_loss):
        return d_reconstr_loss - self.k * g_reconstr_loss


    def update_k(self, d_reconstr_loss, g_reconstr_loss):
        self.k += (self.lmbda * \
            (self.gamma * d_reconstr_loss - g_reconstr_loss))
        self.k = torch.clamp(self.k, min=0)
        self.k = self.k.detach()


    def reconstr_loss(self, images, num_images):
        r_images = self.discriminator(images)
        return (1 / (num_images*self.img_size*self.img_size)) * \
            self.L1_norm(r_images, images)


    def L1_norm(self, x, y):
        return torch.tensor((x - y).abs().sum())


    def set_optimizers(self, discrim_optimizer, gen_optimizer):
        self.discrim_optimizer = discrim_optimizer
        self.gen_optimizer = gen_optimizer


    def train_batch(self, images):
        # NCHW
        num_images = images.size()[0]

        d_reconstr_loss = self.reconstr_loss(images, num_images)

        noise = torch.rand(num_images, 3, self.img_size, self.img_size).to(self.device)
        fake_images = self.generator(noise)
        gen_loss = self.reconstr_loss(fake_images, num_images)

        discrim_loss = self.discriminator_loss(
            d_reconstr_loss, 
            g_reconstr_loss, 
        )

        self.discriminator.zero_grad()
        discrim_loss.backward(retain_graph=True)
        self.discrim_optimizer.step()

        self.generator.zero_grad()
        gen_loss.backward()
        self.gen_optimizer.step()

        self.update_k(d_reconstr_loss, g_reconstr_loss)

        self.batch_conv_measure = ( 
            d_reconstr_loss + \
            (self.gamma * d_reconstr_loss - g_reconstr_loss).abs()
        ).detach()

        self.batch_d_reconstr_loss = d_reconstr_loss.detach()
        self.batch_g_reconstr_loss = g_reconstr_loss.detach()

