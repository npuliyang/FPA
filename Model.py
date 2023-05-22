#!/usr/bin/env python 
import torch.nn as nn
import torch 
from torch.autograd import Variable
import torch.nn.functional as F
import random
import numpy as np 
import torchvision.models as models
class CNN_base(nn.Module):
    def __init__(self):
        super(CNN_base, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(-1, 320)
        x = self.fc1(x)
        x = F.relu(x)

        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


class WhiteAttack(nn.Module):
    def __init__(self, args):
        super(WhiteAttack, self).__init__()
        if args.image =='gray':
            self.pert_att = pert_Att_GrayScale()
        else:
            self.pert_att = pert_Att(args) 
        self.args = args 
        # self.modela = model
    def forward(self, x, model):
        # self.model = model
        # self.model.eval()
        # model_new = model.deepcopy()
        if self.args.image == 'rgb':
            x = x/255.0 
        x = model(x)
        return x
    def perturb(self, x):
        x_ori = x.clone()
        # x_data = x_ori.clone().detach()
        batch_size = x.size()[0]
         
        if self.args.image == 'rgb':
            x_pert_ori, rec, p, xs = self.pert_att(x_ori/255.0)
        else:
            x_pert_ori, rec, p, xs = self.pert_att(x_ori)
            
        x_view = xs.view(batch_size, -1)
        topk = torch.topk(x_view, 15)[0][:,-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        
        mask = Variable(torch.ones_like(x), requires_grad=True)  
        mask = mask * topk
        
        mask = torch.gt(xs, mask) * rec
        
        x_pert = x_pert_ori * mask 
        x_ori_th =  x_ori * mask 
        
        x_pert_mi = x_pert + x_ori_th
        
        if self.args.image == 'rgb':
            x_pert_0 = torch.clamp(x_pert_mi, min=0.00, max = 255.0)
        else:
            x_pert_0 = torch.clamp(x_pert_mi, min=0.00, max = 1.0) 
            
        x_pert_ = x_pert_0 - x_ori_th
        
        return x_pert_, x_pert_ori, rec, p

class AttackNN(nn.Module):
    def __init__(self, args):
        super(AttackNN, self).__init__()
        self.args = args
        if args.image == 'gray':
            inchannel = 1
            self.pert_att = pert_Att_GrayScale()
            self.feature_num = 320
        else:
            inchannel = 3
            self.pert_att = pert_Att(args)
            self.feature_num = 500
        
        self.conv1 = nn.Conv2d(inchannel, 10, kernel_size=5)
        # self.MSELoss = nn.MSELoss()
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        
        if args.dataset =='cifar100':
            self.fc1 = nn.Linear(self.feature_num, 500)
            self.fc2 = nn.Linear(500, 100)
        elif args.dataset =='imagenet':
            self.pert_att = pert_Att_IMN(args)

            self.fc1 = nn.Linear(self.feature_num, 2000)
            self.fc2 = nn.Linear(2000, 1000)
        else:
            self.fc1 = nn.Linear(self.feature_num, 50)
            self.fc2 = nn.Linear(50, 10)



        
    def forward(self, x):

        if self.args.image == 'rgb':
            x = x/255.0
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = x.view(-1, self.feature_num)
        x = self.fc1(x)
        x = F.relu(x)

        x = F.dropout(x, training=self.training)
        x = F.relu(self.fc2(x))
        
        return x

    def perturb(self, x):
        x_ori = x.clone()
        # x_data = x_ori.clone().detach()
        batch_size = x.size()[0]
        if self.args.image == 'rgb':
            x_pert_ori, rec, p, xs = self.pert_att(x_ori/255.0)
        else:
            x_pert_ori, rec, p, xs = self.pert_att(x_ori)


        x_view = xs.view(batch_size, -1)
        topk = torch.topk(x_view, 16)[0][:,-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        mask = Variable(torch.ones_like(x), requires_grad=True)  
        mask = mask * topk

        mask = torch.gt(xs, mask) * rec

        x_pert = x_pert_ori * mask 
        x_ori_th =  x_ori * mask 

        x_pert_mi = x_pert + x_ori_th
        
        if self.args.image == 'rgb':
            x_pert_0 = torch.clamp(x_pert_mi, min=0.00, max = 255.0)
        else:
            x_pert_0 = torch.clamp(x_pert_mi, min=0.00, max = 1.0) 
 
        x_pert_ = x_pert_0 - x_ori_th
         
        return x_pert_, x_pert_ori, rec, p
# define a Conv VAE
class AttackNN_blackbox(nn.Module):
    def __init__(self, args):
        super(AttackNN_blackbox, self).__init__()
        if args.image == 'gray':
            inchannel = 1
            self.feature_num = 320
            self.pert_att = pert_Att_GrayScale()
        elif args.image =='rgb':
            inchannel = 3
            self.feature_num = 500
            self.pert_att = pert_Att(args)
        self.conv1 = nn.Conv2d(inchannel, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(self.feature_num, 50)
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = F.relu(x) 
        x = x.view(-1, self.feature_num)
        x = self.fc1(x)
        x = F.relu(x)

        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x  

    def perturb(self, x):
        x_ori = x.clone()
        # x_data = x_ori.clone().detach()
        batch_size = x.size()[0]
   
        x_pert_ori, rec, p, xs = self.pert_att(x_ori/255.0)


        x_view = xs.view(batch_size, -1)
        topk = torch.topk(x_view, 101)[0][:,-1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
 
        mask = Variable(torch.ones_like(x), requires_grad=True)  
        mask = mask * topk


        mask = torch.gt(xs, mask) * rec
        
        x_pert = x_pert_ori #*rec#* mask 
        x_ori_th =  x_ori# * mask
        
        x_pert = torch.clamp(x_pert, min = -5.0, max = 5.0) 
        x_pert_mi = x_pert +  x_ori_th
        
        x_pert_0 = torch.clamp(x_pert_mi, min=0.00, max = 255.0)
        
        x_pert_ = x_pert_0 - x_ori_th

        return x_pert_, x_pert_ori, rec, p
class pert_Att_IMN(nn.Module):
    def __init__(self, args):
        super(pert_Att_IMN, self).__init__()
 
        # encoder
        image_channels = 3
        init_channels = 8
        kernel_size = 4
        latent_dim = 16
        self.leaky = nn.LeakyReLU() 
        self.enc0 = nn.Conv2d(in_channels = image_channels, out_channels=1, kernel_size=1)
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BN1 = nn.BatchNorm2d(init_channels)
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BN2 = nn.BatchNorm2d(init_channels*2)
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BN3 = nn.BatchNorm2d(init_channels*4)
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, 
            stride=1, padding=1
        )
        self.BN4 = nn.BatchNorm2d(64)
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        self.fc21 = nn.Linear(latent_dim, 64)
        
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        self.BNd1 = nn.BatchNorm2d(init_channels*8)
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd2 = nn.BatchNorm2d(init_channels*4)
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd3 = nn.BatchNorm2d(init_channels*2)
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd4 = nn.BatchNorm2d(image_channels)
        self.dec11 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        self.BNd11 = nn.BatchNorm2d(init_channels*8)
        self.dec21 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd21 = nn.BatchNorm2d(init_channels*4)
        self.dec31 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd31 = nn.BatchNorm2d(init_channels*2)
        self.dec41 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd41 = nn.BatchNorm2d(image_channels)
        # self.batchnormal = torch.nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):

        x = self.leaky(self.BN1(self.enc1(x)))

        x = self.leaky(self.BN2(self.enc2(x)))

        x = self.leaky(self.BN3(self.enc3(x)))

        x = self.leaky((self.enc4(x)))

        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)

        log_var = self.fc_log_var(hidden)

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z0 = self.fc2(z)
        z1 = self.fc21(z) 
        z0 = z0.view(-1, 64, 1, 1)
       
        z1 = z1.view(-1, 64, 1, 1)
 
        # decoding
        x = (self.BNd1(self.dec1(z0)))

        x =  self.BNd2((self.dec2(x)))

        x =  (self.leaky(self.dec3(x)))
        pert =  (self.leaky(self.dec4(x)))
        
        xs = (self.BNd11(self.dec11(z1)))
        xs = self.BNd21((self.dec21(xs)))
        xs =  (self.leaky(self.dec31(xs)))
        xs =  (self.leaky(self.dec41(xs)))
        rec = F.sigmoid(xs)


        return pert, rec, (mu, log_var), xs 

class pert_Att(nn.Module):
    def __init__(self, args):
        super(pert_Att, self).__init__()
 
        # encoder
        image_channels = 3
        init_channels = 8
        kernel_size = 4
        latent_dim = 16
        self.leaky = nn.LeakyReLU() 
        self.enc0 = nn.Conv2d(in_channels = image_channels, out_channels=1, kernel_size=1)
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BN1 = nn.BatchNorm2d(init_channels)
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BN2 = nn.BatchNorm2d(init_channels*2)
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BN3 = nn.BatchNorm2d(init_channels*4)
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, 
            stride=1, padding=1
        )
        self.BN4 = nn.BatchNorm2d(64)
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        self.fc21 = nn.Linear(latent_dim, 64)
        
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        self.BNd1 = nn.BatchNorm2d(init_channels*8)
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd2 = nn.BatchNorm2d(init_channels*4)
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd3 = nn.BatchNorm2d(init_channels*2)
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd4 = nn.BatchNorm2d(image_channels)
        self.dec11 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        self.BNd11 = nn.BatchNorm2d(init_channels*8)
        self.dec21 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd21 = nn.BatchNorm2d(init_channels*4)
        self.dec31 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd31 = nn.BatchNorm2d(init_channels*2)
        self.dec41 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd41 = nn.BatchNorm2d(image_channels)


    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
   
        x = self.leaky(self.BN1(self.enc1(x)))

        x = self.leaky(self.BN2(self.enc2(x)))

        x = self.leaky(self.BN3(self.enc3(x)))

        x = self.leaky((self.enc4(x)))
        

        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)

        log_var = self.fc_log_var(hidden)

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z0 = self.fc2(z)
        z1 = self.fc21(z)
        z0 = z0.view(-1, 64, 1, 1)
       
        z1 = z1.view(-1, 64, 1, 1)
 
        # decoding
        x = (self.BNd1(self.dec1(z0)))

 

        x =  self.BNd2((self.dec2(x)))
        
        x =  (self.leaky(self.dec3(x)))
        
        pert =  (self.leaky(self.dec4(x)))
        
        xs = (self.BNd11(self.dec11(z1)))
        xs = self.BNd21((self.dec21(xs)))
        xs =  (self.leaky(self.dec31(xs)))
        xs =  (self.leaky(self.dec41(xs)))
        
        rec = F.sigmoid(xs)

        return pert, rec, (mu, log_var), xs


class pert_Att_GrayScale(nn.Module):
    def __init__(self):
        super(pert_Att_GrayScale, self).__init__()
 
        # encoder
        image_channels = 1
        init_channels = 8
        kernel_size = 4
        latent_dim = 16
        self.enc1 = nn.Conv2d(
            in_channels=image_channels, out_channels=init_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BN1 = nn.BatchNorm2d(init_channels)
        self.enc2 = nn.Conv2d(
            in_channels=init_channels, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BN2 = nn.BatchNorm2d(init_channels*2)
        self.enc3 = nn.Conv2d(
            in_channels=init_channels*2, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BN3 = nn.BatchNorm2d(init_channels*4)
        self.enc4 = nn.Conv2d(
            in_channels=init_channels*4, out_channels=64, kernel_size=kernel_size, 
            stride=1, padding=1
        )
        self.BN4 = nn.BatchNorm2d(64)
        # fully connected layers for learning representations
        self.fc1 = nn.Linear(64, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_log_var = nn.Linear(128, latent_dim)
        self.fc2 = nn.Linear(latent_dim, 64)
        # decoder 
        self.dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=init_channels*8, kernel_size=kernel_size, 
            stride=2, padding=0
        )
        self.BNd1 = nn.BatchNorm2d(init_channels*8)
        self.dec2 = nn.ConvTranspose2d(
            in_channels=init_channels*8, out_channels=init_channels*4, kernel_size=kernel_size, 
            stride=1, padding=0
        )
        self.BNd2 = nn.BatchNorm2d(init_channels*4)
        self.dec3 = nn.ConvTranspose2d(
            in_channels=init_channels*4, out_channels=init_channels*2, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd3 = nn.BatchNorm2d(init_channels*2)
        self.dec4 = nn.ConvTranspose2d(
            in_channels=init_channels*2, out_channels=image_channels, kernel_size=kernel_size, 
            stride=2, padding=1
        )
        self.BNd4 = nn.BatchNorm2d(image_channels)
        # self.batchnormal = torch.nn.BatchNorm2d(1, eps=1e-05, momentum=0.1)

    def reparameterize(self, mu, log_var):
        """
        :param mu: mean from the encoder's latent space
        :param log_var: log variance from the encoder's latent space
        """
        std = torch.exp(0.5*log_var) # standard deviation
        eps = torch.randn_like(std) # `randn_like` as we need the same size
        sample = mu + (eps * std) # sampling
        return sample
 
    def forward(self, x):
        # encoding
        
        x = F.relu(self.BN1(self.enc1(x)))
        x = F.relu(self.BN2(self.enc2(x)))

        x = F.relu(self.BN3(self.enc3(x)))

        x = F.relu(self.enc4(x))

        batch, _, _, _ = x.shape
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        hidden = self.fc1(x)
        # get `mu` and `log_var`
        mu = self.fc_mu(hidden)

        log_var = self.fc_log_var(hidden)

        # get the latent vector through reparameterization
        z = self.reparameterize(mu, log_var)
        z = self.fc2(z)
        z = z.view(-1, 64, 1, 1)
 
        # decoding
        x = (self.BNd1(self.dec1(z)))
     
        x = (self.BNd2(self.dec2(x)))


        x =  ((self.dec3(x)))

        pert =  (self.dec4(x))
        rec = F.sigmoid(pert)

        return pert, rec, (mu, log_var), rec
 
class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        self.fc = nn.Linear(2048, 128)
        
        self.out = nn.Linear(128, 10) 
    
    def forward(self, x):
        x = F.leaky_relu(self.fc(x))
        
        # branch a
        a = F.leaky_relu(self.out(x))
        
        
        return a

class CNN_stat(nn.Module):
    """docstring for CNN_stat"""
    def __init__(self):
        super(CNN_stat, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        # print(x.size())
        x = self.conv1(x)
        self.x1 = (torch.mean(x), torch.std(x))
        x = F.max_pool2d(x, 2)
        self.x2 = (torch.mean(x), torch.std(x))
        x = F.relu(x)
        self.x3 = (torch.mean(x), torch.std(x))

        x = self.conv2(x)
        self.x4 = (torch.mean(x), torch.std(x))

        x = self.conv2_drop(x)
        self.x5 = (torch.mean(x), torch.std(x))

        x = F.max_pool2d(x, 2)
        self.x6 = (torch.mean(x), torch.std(x))

        x = F.relu(x)
        self.x7 = (torch.mean(x), torch.std(x))

        x = x.view(-1, 320)
        x = self.fc1(x)
        self.x8 = (torch.mean(x), torch.std(x))

        x = F.relu(x)
        self.x9 = (torch.mean(x), torch.std(x))

        x = F.dropout(x, training=self.training)
        self.x10 = (torch.mean(x), torch.std(x))

        x = self.fc2(x)
        self.x11 = (torch.mean(x), torch.std(x))

        return F.log_softmax(x)

# Generator Code

class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed

class VGG_GrayScale(nn.Module):
    def __init__(self, class_dim):
        super(VGG_GrayScale, self).__init__()

        vgg_firstlayer=models.vgg16(pretrained = True).features[0] #load just the first conv layer
        vgg=models.vgg16(pretrained = True).features[1:30] #load upto the classification layers except first conv layer

        w1=vgg_firstlayer.state_dict()['weight'][:,0,:,:]
        w2=vgg_firstlayer.state_dict()['weight'][:,1,:,:]
        w3=vgg_firstlayer.state_dict()['weight'][:,2,:,:]
        w4=w1+w2+w3 # add the three weigths of the channels
        w4=w4.unsqueeze(1)# make it 4 dimensional


        first_conv=nn.Conv2d(1, 64, 3, padding = (1,1)) #create a new conv layer
        first_conv.weigth=torch.nn.Parameter(w4, requires_grad=True) #initialize  the conv layer's weigths with w4
        first_conv.bias=torch.nn.Parameter(vgg_firstlayer.state_dict()['bias'], requires_grad=True) #initialize  the conv layer's weigths with vgg's first conv bias


        self.first_convlayer=first_conv #the first layer is 1 channel (Grayscale) conv  layer
        self.vgg =nn.Sequential(vgg)

        self.fc1 = nn.Linear(512, 1000)
        self.fc2 = nn.Linear(1000, class_dim)

    def forward(self, x):
    
        x=self.first_convlayer(x)
   
        x = self.vgg(x)
        x = x.view(-1, 512)
        x = F.relu(self.fc1(x))

        x = self.fc2(x)
        return x

class ResNet_GrayScale(nn.Module):
    def __init__(self, class_dim):
        super(ResNet_GrayScale, self).__init__()

        resnet_firstlayer=models.resnet50(pretrained = True).conv1 #load just the first conv layer
        resnet=models.resnet50(pretrained = True) #load upto the classification layers except first conv layer
         

        w1=resnet_firstlayer.state_dict()['weight'][:,0,:,:]
        w2=resnet_firstlayer.state_dict()['weight'][:,1,:,:]
        w3=resnet_firstlayer.state_dict()['weight'][:,2,:,:]
        w4=w1+w2+w3 # add the three weigths of the channels
        w4=w4.unsqueeze(1)# make it 4 dimensional


        first_conv=nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False) #create a new conv layer
        first_conv.weigth=torch.nn.Parameter(w4, requires_grad=True) #initialize  the conv layer's  
        resnet.conv1 = first_conv
 
        num_features = resnet.fc.in_features
        # features = nn.Linear(num_features, class_dims) # Add our layer with 4 outputs
        resnet.fc =  nn.Linear(num_features, class_dim)
        self.resnet = resnet

    def forward(self, x):
        
        x = self.resnet(x)
        
        return x