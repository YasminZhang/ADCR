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



###
###
### Based on https://arxiv.org/abs/1805.11572
###
###

# write a convexnet with MLPs instead of CNNs
class MLP_convexnet(nn.Module):
    def __init__(self, args, n_channels=200, n_layers=10,n_chan=128):
        super().__init__()
        self.args=args
        self.convex = args.wclip
        self.n_layers = n_layers
        self.leaky_relu = nn.LeakyReLU(negative_slope=.2)
        self.smooth_length=0
        # these layers can have arbitrary weights
        self.wxs = nn.ModuleList([nn.Linear(n_chan, n_channels) for _ in range(self.n_layers+1)])
        # these layers should have non-negative weights
        self.wzs = nn.ModuleList([nn.Linear(n_channels, n_channels) for _ in range(self.n_layers)])
        self.final_conv2d = nn.Linear(n_channels, 1)
        # self.initialize_weights()

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


#ICNN
class convexnet(nn.Module):
    def __init__(self, args, n_channels=16, kernel_size=5, n_layers=5, convex=True,n_chan=1):
        super().__init__()
        self.args=args
        self.convex = True
        self.n_layers = n_layers
        self.leaky_relu = nn.LeakyReLU(negative_slope=.2)
        self.smooth_length=0
        # these layers can have arbitrary weights
        self.wxs = nn.ModuleList([nn.Conv2d(n_chan, n_channels, kernel_size=kernel_size, stride=1, padding=2, bias=True) for _ in range(self.n_layers+1)])

        # these layers should have non-negative weights
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

    def wei_dec(self):
        return 0


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

class AR_lim_MLP(nn.Module):
    def __init__(self,args,n_chan=2):
        super(AR_lim_MLP, self).__init__()
        self.act=nn.SiLU#nn.LeakyReLU   
        self.args = args 
        self.convnet = nn.Sequential(
            nn.Linear(n_chan, 128,  ),
            self.act(),
            nn.Linear(128, 128,  ),
            self.act(),
            nn.Linear(128, 128,  ),
            self.act(),
        )       
    def wei_dec(self):
        return 0


    def init_weights(self,m):
        pass

    def forward(self, image):
        output = self.convnet(image)
        return output



    
##AR architecture
class AR(nn.Module):
    def __init__(self,args,n_chan=1):
        super(AR, self).__init__()
        self.act=nn.SiLU#nn.LeakyReLU   
        self.args = args 
        self.convnet = nn.Sequential(
            nn.Conv2d(n_chan, 16, kernel_size=(5, 5),padding=2),
            self.act(),
            nn.Conv2d(16, 32, kernel_size=(5, 5),padding=2),
            self.act(),
            nn.Conv2d(32, 32, kernel_size=(5, 5),padding=2,stride=2),
            self.act(),
            nn.Conv2d(32, 64, kernel_size=(5, 5),padding=2,stride=2),
            self.act(),
            nn.Conv2d(64, 64, kernel_size=(5, 5),padding=2,stride=2),
            self.act(),
            nn.Conv2d(64, 128, kernel_size=(5, 5),padding=2,stride=2),
            self.act()
        )

        # self.fc = nn.Sequential(
        #     nn.Linear(128*(config.size//16)**2, 256),
        #     nn.LeakyReLU(),
        #     nn.Linear(256, 1)
        # )
    def wei_dec(self):
        return 0
        rate=0.0#1#10#500#10
        for i in range(6):
            self.convnet[2*i].weight.data=(1-2*rate*self.args.lr)*self.convnet[2*i].weight.data


    def init_weights(self,m):
        pass

    def forward(self, image):
        output = self.convnet(image)
        # output = output.view(image.size(0), -1)
        # output = self.fc(output)
        return output
    
class AR_lim(nn.Module):
    def __init__(self,args,n_chan=1):
        super(AR_lim, self).__init__()
        self.act=nn.SiLU#nn.LeakyReLU   
        self.args = args 
        self.convnet = nn.Sequential(
            nn.Conv2d(n_chan, 128, kernel_size=(5, 5),padding=2),
            self.act(),
        )       
    def wei_dec(self):
        return 0


    def init_weights(self,m):
        pass

    def forward(self, image):
        output = self.convnet(image)
        return output

## Architecture I proposed in the initial rpoject report
class MyNet(nn.Module):
    def __init__(self,args,n_chan=1):
        super(MyNet, self).__init__()
        if(args.setting == 'limited'):
            self.smooth=AR_lim(args)
        elif(args.setting == 'sparse'):
            self.smooth=AR(args)
        elif args.setting == 'synthetic':
            self.smooth=AR_lim_MLP(args)

        self.args=args
        self.convex = args.wclip
        if not args.synthetic:
            if args.setting == 'limited':
                self.convnet=convexnet(args,n_channels=16,n_chan=128, n_layers=5)
            elif args.setting == 'sparse':
                self.convnet=convexnet(args,n_channels=128,n_chan=128, n_layers=5)
            else:
                raise ValueError('Setting not recognized')
        else:
            self.convnet=MLP_convexnet(args)
        self.l2_penalty = nn.Parameter((-9.0) * torch.ones(1))

        self.synthetic = args.synthetic


    def clamp_weights(self):
        self.convnet.clamp_weights()
 
    
    def strong_convex_term(self,x):
        l2_term = torch.sum(x.view(x.size(0), -1)**2, dim=1,keepdim=True)
        return  (F.softplus(self.l2_penalty))*l2_term

    def forward(self, image):
        # print(self.l2_penalty)
        # print(self.convnet(self.smooth(image)),self.strong_convex_term(image))
        if not self.synthetic:
            output = self.convnet(self.smooth(image)) + self.strong_convex_term(image)
        else:
            output = self.convnet(self.smooth(image))
        return output

### The network with name MyNet is used by default
class Algorithm(BaseAlg.baseNet):
    def __init__(self,args,data_loaders,path=config.data_path+'nets/'):
        if not args.synthetic:
            if(config.angles==0):
                n_chan=3
            else: n_chan=1
        else:
            n_chan=1
         
        super(Algorithm, self).__init__(args,path,MyNet(args,n_chan=n_chan),data_loaders)
     

 
        self.eps=args.eps
        self.expir=args.expir
        self.mu=args.mu # 0.1 first -> 10
        if not args.synthetic:
            self.lamb=self.lamb_approx()
        else:
            self.lamb=0
        self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=args.lr)
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
        if not self.args.synthetic:
            if(config.angles != 0):fake_samples=config.fbp_op_mod(scans)
            else:fake_samples=scans.clone()
        else:
            fake_samples=scans.clone()
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
        loss = self.net(real_samples).mean()-self.net(fake_samples).mean()+self.mu*((F.relu(gradients.norm(2, dim=1) - 1)) ** 2).mean()
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
            if(c =='d'):
                for g in optimizer.param_groups:
                    g['lr'] = 0.5*g['lr']

            optimizer.zero_grad()
            lossm=lambdas*self.net(guess).sum()
            lossm.backward()

            guess.grad+=grad
            # guess.grad+=grad/torch.norm(grad)
            if(truth is not None):
                gradsizel1=torch.norm(guess.grad,p=1)
                gradsizel2=torch.norm(guess.grad,p=2)
                gradsizel2_fidelity=torch.norm(guess.grad-grad,p=2)
                gradsizel2_regulariser=torch.norm(grad,p=2)
                writer.add_scalar('Grad_size_ratio',gradsizel2_fidelity.item()/gradsizel2_regulariser.item(),j)
                writer.add_scalar('Grad_size_l2',gradsizel2.item(),j)
                writer.add_scalar('Grad_size_l2_fidelity',gradsizel2_fidelity.item(),j)
                writer.add_scalar('Grad_size_l2_regulariser',gradsizel2_regulariser.item(),j)
            optimizer.step()
        writer.close()
        return guess

    def train(self,writer,epoch):
        self.net.train()
        if not os.path.exists(config.data_path+'nets_new/' +self.args.alg + '/'+str(self.args.setting)):
            os.makedirs(config.data_path+'nets_new/' +self.args.alg + '/'+str(self.args.setting))
        for i, ((scans_1,truth_1), (_,truth_2)) in enumerate(zip(self.data_train_loader, self.data_train_loader)):
            if (scans_1.nelement() == 0):
                scans_1 = create(truth_1,self.noisemean)
            start = time.time()

            loss = self.train_one_batch(scans_1,truth_1,writer)
            # self.net.wei_dec()

            end = time.time()

            writer.add_scalar('Loss', loss, (epoch-1)*len(self.data_train_loader)+i)
            if i % self.args.log_interval == 0:
                print(str(self.args.expir) + ':Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Time: {:.6f}s'.format(epoch, i * len(scans_1), len(self.data_train_loader.dataset),100. * i / len(self.data_train_loader), loss,end-start))
                with open(config.data_path+'logs/'+self.args.alg+str(self.args.expir)+'.txt',"a") as f:
                    f.write('Train Epoch: {} [{}/{} ({:.0f}%)] Loss: {:.6f}, Time: {:.6f}s\n'.format(epoch, i * len(scans_1), len(self.data_train_loader.dataset),100. * i / len(self.data_train_loader), loss,end-start))
            # if i % self.args.cp_interval == 0: self.save_checkpoint(config.data_path+'nets/net_cps/net'+self.args.alg+str(self.args.setup)+'ep'+str(epoch)+'no'+str(i)+'exp'+str(self.args.expir)+'.pt')
            if i % self.args.cp_interval == 0: self.save_checkpoint(config.data_path+'nets_new/' +self.args.alg + '/'+str(self.args.setting) + '/'+'ep'+str(epoch)+'no'+str(i)+'exp'+str(self.args.expir)+'.pt')
