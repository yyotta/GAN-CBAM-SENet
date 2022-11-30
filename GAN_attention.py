import utils, torch, time, os, pickle
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import dataloader
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.mlp = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=True),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=True))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
     
    
class SENet(nn.Module):
    def __init__(self, in_chnls, ratio):
        super(SENet, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d((1, 1))
        self.compress = nn.Conv2d(in_chnls, in_chnls//ratio, 1, 1, 0)
        self.excitation = nn.Conv2d(in_chnls//ratio, in_chnls, 1, 1, 0)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.compress(out)
        out = F.relu(out)
        out = self.excitation(out)
        return F.sigmoid(out)
    
    
class generator(nn.Module):
    def __init__(self, input_dim=100, output_dim=1, input_size=32):
        super(generator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.fc = nn.Sequential(
            nn.Linear(self.input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Linear(1024, 128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.BatchNorm1d(128 * (self.input_size // 4) * (self.input_size // 4)),
            nn.ReLU(),
        )
        self.deconv_head = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.preprocess = nn.Sequential(
            nn.AdaptiveAvgPool2d((56, 56))
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.AdaptiveAvgPool2d((56 ,56)),
            nn.ReLU(),          
        )
        self.deconv_tail = nn.Sequential(
            nn.ConvTranspose2d(80, self.output_dim, 4, 2, 1),
            nn.AdaptiveAvgPool2d((96, 96)),
            nn.Tanh(),
        )
        self.ca = ChannelAttention(in_planes=80)
        self.sa = SpatialAttention()
        utils.init_weight(self)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 128, (self.input_size // 4), (self.input_size // 4))
        x = self.deconv_head(x)
        x_preprocess = self.preprocess(x)
        x = self.deconv(x)
        x = torch.cat((x_preprocess, x), dim=1)

        x = self.ca(x) * x
        x = self.sa(x) * x
        x = self.deconv_tail(x)

        return x

class discriminator(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, input_size=32):
        super(discriminator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_size = input_size

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
        )
        self.senet = SENet(in_chnls=128, ratio=16)

        self.fc = nn.Sequential(
            nn.Linear(128 * (self.input_size // 4) * (self.input_size // 4), 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, self.output_dim),
            nn.Sigmoid(),
        )
        utils.init_weight(self)

    def forward(self, x):
        x = self.conv(x)

        coefficient = self.senet(x)
        x = coefficient * x
        
        x = x.view(-1, 128 * (self.input_size // 4) * (self.input_size // 4))
        x = self.fc(x)

        return x


class GAN_attention(object):
    def __init__(self, args):
        # parameters
        self.epoch = args.epoch
        self.sample_num = 100
        self.batch_size = args.batch_size
        self.save_dir = args.save_dir
        self.result_dir = args.result_dir
        self.dataset = args.dataset
        self.log_dir = args.log_dir
        self.use_cuda = args.use_cuda
        self.gan_type = args.gan_type
        self.input_size = args.input_size
        self.print_frequency = args.print_frequency
        self.z_dim = 62

        # load dataset
        self.data_loader = dataloader(self.dataset, self.input_size, self.batch_size)
        # The dimension of output img generated, which is the same as imgs in dataset.
        img_dim = self.data_loader.__iter__().__next__()[0].shape[1]

        # networks init
        self.G = generator(input_dim=self.z_dim, output_dim=img_dim, input_size=self.input_size)
        self.D = discriminator(input_dim=img_dim, output_dim=1, input_size=self.input_size)
        self.G_optimizer = optim.Adam(self.G.parameters(), lr=args.lr_g, betas=(0.5, 0.999))
        self.D_optimizer = optim.Adam(self.D.parameters(), lr=args.lr_d, betas=(0.5, 0.999))

        if self.use_cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device =  torch.device('cpu')
            
        self.G.to(self.device)
        self.D.to(self.device)
        self.BCE_loss = nn.BCELoss().to(self.device)

        print('---------- Networks architecture -------------')
        utils.print_network(self.G)
        utils.print_network(self.D)
        print('-----------------------------------------------')
        print('Using device : ', self.device)


        # fixed noise
        self.sample_z = torch.rand((self.batch_size, self.z_dim))
        self.sample_z = self.sample_z.to(self.device)


    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['epoch_using_time'] = []
        self.train_hist['total_time'] = []

        self.y_real, self.y_fake = torch.ones(self.batch_size, 1), torch.zeros(self.batch_size, 1)
        self.y_real, self.y_fake = self.y_real.to(self.device), self.y_fake.to(self.device)

        self.D.train()
        print('training start!!')
        start_time = time.time()
        for epoch in range(self.epoch):
            self.G.train()
            epoch_start_time = time.time()
            for iteration, (x, _) in enumerate(self.data_loader):
                if iteration == self.data_loader.dataset.__len__() // self.batch_size:
                    break

                z = torch.rand((self.batch_size, self.z_dim))
                if self.use_cuda:
                    x, z = x.to(self.device), z.to(self.device)

                # update D network
                self.D_optimizer.zero_grad()

                D_real = self.D(x)
                D_real_loss = self.BCE_loss(D_real, self.y_real)

                data_generated = self.G(z)
                D_fake = self.D(data_generated)
                D_fake_loss = self.BCE_loss(D_fake, self.y_fake)

                D_loss = D_real_loss + D_fake_loss
                self.train_hist['D_loss'].append(D_loss.item())

                D_loss.backward()
                self.D_optimizer.step()

                # update G network
                self.G_optimizer.zero_grad()

                data_generated = self.G(z)
                D_fake = self.D(data_generated)
                G_loss = self.BCE_loss(D_fake, self.y_real)
                self.train_hist['G_loss'].append(G_loss.item())

                G_loss.backward()
                self.G_optimizer.step()

                if ((iteration + 1) % self.print_frequency) == 0:
                    print('Epoch : [{}] [{}/{}] | D_loss : {} | G_loss : {}'.format(epoch+1, iteration+1, 
                            self.data_loader.dataset.__len__() // self.batch_size, round(D_loss.item(), 6), round(G_loss.item(), 6)))

            self.train_hist['epoch_using_time'].append(time.time() - epoch_start_time)
            with torch.no_grad():
                self.save_imgs((epoch+1))

        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['epoch_using_time']),
              self.epoch, self.train_hist['total_time'][0]))
        print("Training finish!... save training results")

        self.save_model()
        utils.generate_animation(os.path.join(self.result_dir, self.dataset, self.gan_type, self.gan_type), self.epoch)
        utils.loss_plot(self.train_hist, os.path.join(self.save_dir, self.dataset, self.gan_type), self.gan_type)

    def save_imgs(self, epoch, fix=True):
        self.G.eval()

        if not os.path.exists(self.result_dir + '/' + self.dataset + '/' + self.gan_type):
            os.makedirs(self.result_dir + '/' + self.dataset + '/' + self.gan_type)

        tot_num_samples = min(self.sample_num, self.batch_size)
        image_frame_dim = int(np.floor(np.sqrt(tot_num_samples)))

        if fix:
            """ fixed noise """
            samples = self.G(self.sample_z)
        else:
            """ random noise """
            sample_z = torch.rand((self.batch_size, self.z_dim))
            if self.use_cuda:
                sample_z = sample_z.cuda()

            samples = self.G(sample_z)

        if self.use_cuda:
            samples = samples.cpu().data.numpy().transpose(0, 2, 3, 1)
        else:
            samples = samples.data.numpy().transpose(0, 2, 3, 1)

        samples = (samples + 1) / 2
        utils.save_images(samples[:image_frame_dim * image_frame_dim, :, :, :], [image_frame_dim, image_frame_dim],
                          self.result_dir + '/' + self.dataset + '/' + self.gan_type + '/' + self.gan_type + '_epoch%03d' % epoch + '.png')

    def save_model(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.gan_type)

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        torch.save(self.G.state_dict(), os.path.join(save_dir, self.gan_type + '_G.pkl'))
        torch.save(self.D.state_dict(), os.path.join(save_dir, self.gan_type + '_D.pkl'))

        with open(os.path.join(save_dir, self.gan_type + '_history.pkl'), 'wb') as f:
            pickle.dump(self.train_hist, f)

    def load(self):
        save_dir = os.path.join(self.save_dir, self.dataset, self.gan_type)

        self.G.load_state_dict(torch.load(os.path.join(save_dir, self.gan_type + '_G.pkl')))
        self.D.load_state_dict(torch.load(os.path.join(save_dir, self.gan_type + '_D.pkl')))

