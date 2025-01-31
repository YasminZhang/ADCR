import odl
import torch
import config
import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import sys
import os
from torch.autograd import Variable
import time
from data_load import create
import BaseAlg
from tensorboardX import SummaryWriter
from odl.contrib import fom
import torch.nn.utils.parametrize as P

def power_method(op, maxiter=100, tol=1e-6):
    arr_old = np.random.rand(*op.domain_shape).astype(np.float32)
    error = tol + 1
    i = 0
    while error >= tol:

        # very verbose and inefficient for now
        omega = op(arr_old)
        alpha = np.linalg.norm(omega)
        u = (1.0 / alpha) * omega
        z = op.T(u)
        beta = np.linalg.norm(z)
        arr = (1.0 / beta) * z
        error = np.linalg.norm(op(arr) - beta * u)
        sigma = beta
        arr_old = arr
        i += 1
        if i >= maxiter:
            return sigma

    return sigma



#ICNN
class convexnet(nn.Module):
    def __init__(self, args, n_channels=16, kernel_size=5, n_layers=10, n_chan=1):
        super().__init__()
        self.args=args
        self.convex = True
        self.n_layers = n_layers
        self.leaky_relu = nn.LeakyReLU(negative_slope=.2)
        self.smooth_length=0
        self.wxs = nn.ModuleList([nn.Conv2d(n_chan, n_channels, kernel_size=kernel_size, stride=1, padding=2, bias=True) for _ in range(self.n_layers+1)])
        self.wzs = nn.ModuleList([nn.Conv2d(n_channels, n_channels, kernel_size=kernel_size, stride=1, padding=2, bias=False) for _ in range(self.n_layers)])
        self.final_conv2d = nn.Conv2d(n_channels, 1, kernel_size=kernel_size, stride=1, padding=2, bias=False)

        self.initialize_weights()

    def initialize_weights(self, min_val=0, max_val=1e-3):
        for layer in range(self.n_layers):
            self.wzs[layer].weight.data = min_val + (max_val - min_val) * torch.rand_like(self.wzs[layer].weight.data)
        self.final_conv2d.weight.data = min_val + (max_val - min_val) * torch.rand_like(self.final_conv2d.weight.data)

    def clamp_weights(self):
        for i in range(self.smooth_length,self.n_layers):
            self.wzs[i].weight.data.clamp_(0)
        self.final_conv2d.weight.data.clamp_(0)

    def forward(self, x, grady=False):
        if self.convex:
            self.clamp_weights()
        z = self.leaky_relu(self.wxs[0](x))
        for layer_idx in range(self.n_layers):
            z = self.leaky_relu(self.wzs[layer_idx](z) + self.wxs[layer_idx+1](x))
        z = self.final_conv2d(z)
        net_output = z.view(z.shape[0], -1).mean(dim=1,keepdim=True)
        assert net_output.shape[0] == x.shape[0], f"{net_output.shape}, {x.shape[0]}"
        return net_output
    

class MyNet(nn.Module):
    def __init__(self,args,n_chan=1, layers=10, n_channels=16):
        super(MyNet, self).__init__()
        self.convnet=convexnet(args,n_channels=n_channels,n_layers=layers,n_chan=n_chan)
        self.l2_penalty = nn.Parameter((-9.0) * torch.ones(1)) # strong convex term, not required for now
        # self.op_norm = power_method(self.op)
        # learning_rate = 1 / (self.op_norm) ** 2
 
    
    def strong_convex_term(self,x):
        l2_term = torch.sum(x.view(x.size(0), -1)**2, dim=1,keepdim=True)
        return  (F.softplus(self.l2_penalty))*l2_term

    def forward(self, image):
        output = self.convnet(image) + self.strong_convex_term(image)
        return output
    


### The network with name MyNet is used by default
class Algorithm(BaseAlg.baseNet):
    def __init__(self,args,data_loaders,path=config.data_path+'nets/'):
        if(config.angles==0):
            n_chan=3
        else: n_chan=1
        
        self.eps=args.eps
        self.expir=args.expir
        
        
        if args.setting == 'sparse':
            super(Algorithm, self).__init__(args,path,MyNet(args,n_chan=n_chan, n_channels=48, layers=10),data_loaders)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=2e-5, betas=(0.5, 0.99))
            self.mu=5
        elif args.setting == 'limited':
            super(Algorithm, self).__init__(args,path,MyNet(args,n_chan=n_chan, n_channels=16, layers=5),data_loaders)
            self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=5e-5)
            self.mu=10
        self.lamb=self.lamb_approx()
        self.nograd=False
        self.cntr=1

    def lamb_approx(self):
        if (len(self.data_valid_loader) != 0):
            for i, (scans, truth) in enumerate(self.data_valid_loader):
                if (scans.nelement() == 0):
                    scans = create(truth,self.noisemean)
                if i ==1: break
        else:
            for i, (scans, truth) in enumerate(self.data_test_loader):
                if (scans.nelement() == 0):
                    scans = create(truth,self.noisemean)
                if i ==1: break
        test_images = Variable(truth)
        test_data = Variable(scans)

        if(config.angles!=0):gradient_truth = config.fwd_op_adj_mod((config.fwd_op_mod(truth)-scans))
        else:gradient_truth=truth-scans
        lambdy = np.mean(np.sqrt(np.sum(np.square(gradient_truth.numpy()), axis=(1, 2, 3))))
        print('Lambda: '+str(lambdy))
        return lambdy

    def grady(self,x):
        a=x.clone().requires_grad_(True)
        fake = Variable(torch.Tensor(a.shape[0], 1).fill_(1.0).type_as(x), requires_grad=False)
        grad = torch.autograd.grad(self.net(a),a,grad_outputs=fake)[0]
        return grad

    def loss(self,scans,truth):
        """Calculates the gradient penalty loss for WGAN GP"""
        if(config.angles != 0):fake_samples=config.fbp_op_mod(scans)
        else:fake_samples=scans.clone()
        real_samples=truth

        alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).type_as(truth)
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        net_interpolates = self.net(interpolates)

        fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0).type_as(truth), requires_grad=False)
        gradients = torch.autograd.grad(
            outputs=net_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        decay_loss=0
        loss = self.net(real_samples).mean()-self.net(fake_samples).mean()+self.mu*(((gradients.norm(2, dim=1) - 1)) ** 2).mean()
        return loss

    def output(self,scans,truth=None,lambd=0):
        eps=self.eps
        if(config.angles != 0):
            guess = config.fbp_op_mod(scans)
        else: guess = scans.clone()
        if(lambd==0):
            lambdas=self.lamb
        else: lambdas=lambd
        grad = torch.zeros(guess.shape).type_as(guess)

        guess=torch.nn.Parameter(guess)
        optimizer = torch.optim.SGD([guess], lr=self.eps, momentum=0.5)
        self.cntr+=1
        prevpsn=0
        curpsn=0
        writer = SummaryWriter(config.data_path+'logs/'+self.args.alg+'/exp'+str(self.args.expir)+'/'+'logger'+str(self.cntr),comment='')
        for j in range(self.args.iterates):

            if(config.angles!=0):
                data_misfit=config.fwd_op_mod(guess)-scans
                grad = config.fwd_op_adj_mod(data_misfit)
            else:
                data_misfit=guess-scans
                grad=data_misfit
            if(truth is not None):
                loss = nn.MSELoss()(guess.detach(),truth.detach().cuda())
                cur_loss = 0
                ssim = self.ssim(guess.detach(),truth.detach())
                psnr = self.psnr(guess.detach(),truth.detach())
                writer.add_scalar('MSE Loss', loss.item(), j)
                writer.add_scalar('SSIM',ssim,j)
                writer.add_scalar('PSNR',psnr,j)
                if(self.args.outp):
                    print(j)
                    print('MSE Loss:', loss.item())
                    print('SSIM:',ssim)
                    print('PSNR:',psnr)
                    if j % 10 == 0:
                        self.save_img(f'Descent:{str(j).zfill(6)}',guess.detach())

                prevpsn=curpsn
                curpsn=psnr
                if(curpsn<prevpsn):
                    writer.close()
                    return guess

            c = config.getch()
            if(c =='q'):
                break
            optimizer.zero_grad()
            lossm=lambdas*self.net(guess).sum()
            lossm.backward()

            guess.grad+=grad

            optimizer.step()
        writer.close()
        return guess

    def train(self,writer,epoch):
        self.net.train()
        # if folder ./data/nets/net_cps not exists, create it
        if not os.path.exists(config.data_path+'nets_new/' +self.args.alg + '/'+str(self.args.setting)):
            os.makedirs(config.data_path+'nets_new/' +self.args.alg + '/'+str(self.args.setting))
        for i, ((scans_1,truth_1), (_,truth_2)) in enumerate(zip(self.data_train_loader, self.data_train_loader)):
            if (scans_1.nelement() == 0):
                scans_1 = create(truth_1,self.noisemean)
            start = time.time()

            loss = self.train_one_batch(scans_1,truth_1,writer)

            end = time.time()

            writer.add_scalar('Loss', loss, (epoch-1)*len(self.data_train_loader)+i)
            if i % self.args.log_interval == 0:
                print(str(self.args.expir) + ':Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Time: {:.6f}s'.format(epoch, i * len(scans_1), len(self.data_train_loader.dataset),100. * i / len(self.data_train_loader), loss,end-start))
                with open(config.data_path+'logs/'+self.args.alg+str(self.args.expir)+'.txt',"a") as f:
                    f.write('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Time: {:.6f}s\n'.format(epoch, i * len(scans_1), len(self.data_train_loader.dataset),100. * i / len(self.data_train_loader), loss,end-start))
            # if i % self.args.cp_interval == 0: self.save_checkpoint(config.data_path+'nets/net_cps/net'+self.args.alg+str(self.args.setup)+'ep'+str(epoch)+'no'+str(i)+'exp'+str(self.args.expir)+'.pt')
            if i % self.args.cp_interval == 0: self.save_checkpoint(config.data_path+'nets_new/' +self.args.alg + '/'+str(self.args.setting) + '/'+'ep'+str(epoch)+'no'+str(i)+'exp'+str(self.args.expir)+'.pt')
