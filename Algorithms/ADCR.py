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
 


# write a convexnet with MLPs instead of CNNs
class MLP_convexnet(nn.Module):
    def __init__(self, args, n_channels=200, n_layers=10,n_chan=2):
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
    def __init__(self, args, n_channels=16, kernel_size=5, n_layers=5, n_chan=1):
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


class convexnet_(nn.Module):
    def __init__(self, args, n_channels=16, kernel_size=5, n_layers=5, n_chan=1):
        super().__init__()
        self.args=args
        self.convex = True
        self.n_layers = n_layers
        self.leaky_relu = nn.LeakyReLU(negative_slope=.2)
        self.smooth_length=0
        # these layers can have arbitrary weights
        channels = [n_channels * 2**i for i in range(n_layers+1)]
        self.wxs = nn.ModuleList([nn.Conv2d(n_chan, channels[i], kernel_size=kernel_size, stride=1, padding=2, bias=True) for i in range(self.n_layers+1)])
        # these layers should have non-negative weights
        self.wzs = nn.ModuleList([nn.Conv2d(channels[i], channels[i+1], kernel_size=kernel_size, stride=1, padding=2, bias=False) for i in range(self.n_layers)])
        self.final_conv2d = nn.Conv2d(channels[-1], 1, kernel_size=kernel_size, stride=1, padding=2, bias=False)

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
    def __init__(self,args,n_chan=1):
        super(MyNet, self).__init__()
        if not args.synthetic:
            if args.setting=='sparse':
                self.convnet2 = convexnet_(args,n_channels=args.n_channels,n_chan=1,n_layers=args.n_layers)
                self.convnet=convexnet_(args,n_channels=args.n_channels ,n_chan=1,n_layers=args.n_layers)
            elif args.setting=='limited':
                self.convnet2 = convexnet(args,n_channels=args.n_channels,n_chan=1,n_layers=args.n_layers)
                self.convnet=convexnet(args,n_channels=args.n_channels ,n_chan=1,n_layers=args.n_layers)
            else:
                raise ValueError('Invalid setting')
        # self.l2_penalty = nn.Parameter((-9.0) * torch.ones(1)) # strong convex term, not required for now
        else:
            self.convnet2 = MLP_convexnet(args,)
            self.convnet=MLP_convexnet(args,)
 
    
    def strong_convex_term(self,x):
        l2_term = torch.sum(x.view(x.size(0), -1)**2, dim=1,keepdim=True)
        return  (F.softplus(self.l2_penalty))*l2_term

    def forward(self, image):
        image = image.to(torch.float32)
         
        output = self.convnet(image) - self.convnet2(image)
        return output
    
    def calculate_net2_grad(self, x):
        
        with torch.enable_grad():
            x.requires_grad = True
            y = self.convnet2(x)
            grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0]
        x.requires_grad = False
        grad = grad.detach()
        return grad   
    
    def approximate_prox_h(self, v, alpha, num_iterations=100, tolerance=1e-6):
        u = v.clone()  # Initialize u
        for _ in range(num_iterations):
            # Compute gradient of the objective function at u
            grad_h = self.compute_subgradient_net1(u)
            
            # Update u using a gradient step
            u_next = u - alpha * grad_h
            
            # Apply any necessary projections or constraints on u_next
            #u_next = project_onto_constraints(u_next)
            
            # Check for convergence
            if torch.norm(u_next - u) < tolerance:
                break
            
            u = u_next
        
        return u
    def compute_subgradient_net1(self, x):
        with torch.enable_grad():
            x.requires_grad = True
            y = self.convnet(x)
            grad = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(y))[0]
        x.requires_grad = False
        grad = grad.detach()
        return grad

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
        self.mu=args.mu
        if args.synthetic:
            self.lamb = 0
        else:
            self.lamb=self.lamb_approx()
        # self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr=args.lr)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=args.lr)
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
        optimizer = torch.optim.SGD([guess], lr=self.eps, momentum=0.5 )
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
                    # if j % 10 == 0:
                    #     self.save_img(f'Descent:{str(j).zfill(6)}',guess.detach())

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


    def output_CCP(self,scans,truth=None,lambd=0):
        eps=self.eps
        if(config.angles != 0):
            guess = config.fbp_op_mod(scans)
        else: guess = scans.clone()
        if(lambd==0):
            lambdas=self.lamb
        else: lambdas=lambd
        grad = torch.zeros(guess.shape).type_as(guess)

        guess=torch.nn.Parameter(guess)
        optimizer = torch.optim.SGD([guess], lr=self.eps, momentum=self.args.momentum)
        self.cntr+=1
        prevpsn=0
        curpsn=0
        writer = SummaryWriter(config.data_path+'logs/'+self.args.alg+'/exp'+str(self.args.expir)+'/'+'logger'+str(self.cntr),comment='')
        for j in range(self.args.iterates):

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
                    # if j % 10 == 0:
                    #     self.save_img(f'Descent:{str(j).zfill(6)}',guess.detach())

                prevpsn=curpsn
                curpsn=psnr
                if(curpsn<prevpsn):
                    writer.close()
                    return guess

            c = config.getch()
            if(c =='q'):
                break
            
            xk = guess.clone().detach()
            temp = self.net.calculate_net2_grad(xk)
            # optimizer = torch.optim.SGD([guess], lr=self.eps, momentum=0.5)
            for k in range(self.args.K):
               

                
                optimizer.zero_grad()
                if(config.angles!=0):
                    data_misfit=config.fwd_op_mod(guess)-scans
                    grad = config.fwd_op_adj_mod(data_misfit)
                else:
                    data_misfit=guess-scans
                    grad=data_misfit
             
                # grad_conv2 = (temp * (guess - xk)).sum(dim=(1,2,3))
                lossm=lambdas*(self.net.convnet(guess)).sum()
                lossm.backward()
                guess.grad+=grad - lambdas* temp

                optimizer.step()
        writer.close()
        return guess
    
    def output_pCCP(self,scans,truth=None,lambd=0):
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
                    # if j % 10 == 0:
                    #     self.save_img(f'Descent:{str(j).zfill(6)}',guess.detach())

                prevpsn=curpsn
                curpsn=psnr
                if(curpsn<prevpsn):
                    writer.close()
                    return guess

            c = config.getch()
            if(c =='q'):
                break
            
            beta = 0.5
            
            xk = guess.clone().detach()
            temp = self.net.calculate_net2_grad(xk)
            yk = xk + beta * (xk - xk_1)
            # optimizer = torch.optim.SGD([guess], lr=self.eps, momentum=0.5)
            for k in range(self.args.K):
                
                optimizer.zero_grad()
                if(config.angles!=0):
                    data_misfit=config.fwd_op_mod(guess)-scans
                    grad = config.fwd_op_adj_mod(data_misfit)
                else:
                    data_misfit=guess-scans
                    grad=data_misfit
             
                # grad_conv2 = (temp * (guess - xk)).sum(dim=(1,2,3))
                lossm=lambdas*(self.net.convnet(guess)).sum()
                lossm.backward()
                guess.grad+=grad - lambdas* temp

                optimizer.step()
        writer.close()
        return guess
    
    def output_PSM(self,scans,truth=None,lambd=0):
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

            if(truth is not None):
                loss = nn.MSELoss()(guess.detach(),truth.detach().cuda())
                cur_loss = 0
                ssim = self.ssim(guess.detach(),truth.detach())
                psnr = self.psnr(guess.detach(),truth.detach())
                writer.add_scalar('MSE Loss', loss.item(), j)
                writer.add_scalar('SSIM',ssim,j)
                writer.add_scalar('PSNR',psnr,j)
                if(True):
                    print(j)
                    print('MSE Loss:', loss.item())
                    print('SSIM:',ssim)
                    print('PSNR:',psnr)
                    # if j % 10 == 0:
                    #     self.save_img(f'Descent:{str(j).zfill(6)}',guess.detach())

                prevpsn=curpsn
                curpsn=psnr
                if(curpsn<prevpsn):
                    writer.close()
                    return guess

            c = config.getch()
            if(c =='q'):
                break

            # optimizer = torch.optim.SGD([guess], lr=self.eps)
            
            
            xk = guess.clone().detach()
            temp = self.net.calculate_net2_grad(xk)
            for k in range(self.args.K):
                optimizer.zero_grad()
                if(config.angles!=0):
                    data_misfit=config.fwd_op_mod(xk)-scans
                    grad = config.fwd_op_adj_mod(data_misfit)
                else:
                    data_misfit=xk-scans
                    grad=data_misfit

                # print config.fwd_op_norm
                # print('Fwd Op Norm: ',config.fwd_op_norm)
                alpha = self.args.alpha_psm 
                proxy = xk - alpha*(grad - lambdas * temp) 
                proxy = proxy.detach().requires_grad_(False)
                lossm=lambdas*(self.net.convnet(guess)).sum()
                b = proxy.shape[0]
                objective = lossm + (1/(2*self.args.alpha_inv)*torch.norm((proxy-guess).reshape(b,-1), dim=1)**2).sum() 
                objective.backward()
                optimizer.step()
        writer.close()
        return guess


    def train(self,writer,epoch):
        self.net.train()
        # if folder ./data/nets/net_cps not exists, create it
        if not os.path.exists(config.data_path+'nets_new/' +self.args.alg + '/'+str(self.args.setting) + '/'+ 'exp'+str(self.args.expir) + '/'):
            os.makedirs(config.data_path+'nets_new/' +self.args.alg + '/'+str(self.args.setting) + '/'+ 'exp'+str(self.args.expir) + '/')
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
        self.save_checkpoint(config.data_path+'nets_new/' +self.args.alg + '/'+str(self.args.setting) + '/'+ 'exp'+str(self.args.expir) + '/' + 'ep'+str(epoch)+'End.pt')


    def validate_cycle(self,epoch,end_res,test,writer=None,hyp_writer=None):
        avg_loss=0.0
        all_ssim=0.0
        all_psnr=0.0
        avg_mse_loss=0.0
        print("Validate cycle of {}".format(self.alg))
        if test: data=self.data_test_loader
        else: data=self.data_valid_loader

        start=time.time()
        for i, (scans, truth) in enumerate(data):
            
            print(f'validate cycle: {i}/{len(data)}' )
            if (scans.nelement() == 0):
                scans = create(truth,self.noisemean)
            if self.args.cuda:
                scans,truth = scans.cuda(), truth.cuda()
            if self.args.test_mode == 'CCP':
                output = self.output_CCP(scans,truth)
            elif self.args.test_mode == 'GD':
                output = self.output(scans,truth)
            elif self.args.test_mode == 'PSM':
                output = self.output_PSM(scans,truth)
            else:
                raise ValueError('Invalid test mode')
            cur_loss = self.loss(scans, truth)
            if (type(cur_loss) is tuple): #Checking that we only have one loss not Multiple
                cur_loss=cur_loss[0]
            cur_loss=cur_loss.detach().cpu().item()
            mse_loss = nn.MSELoss()(output,truth).detach().cpu().item()

            avg_ssim = self.ssim(output,truth)
            avg_psnr = self.psnr(output,truth)

            avg_loss+= cur_loss
            avg_mse_loss+=mse_loss
            all_ssim+=avg_ssim
            all_psnr+=avg_psnr
            if (writer is not None):
                writer.add_scalar('SSIM',avg_ssim,i, (epoch-1)*len(data)+i)
                writer.add_scalar('PSNR',avg_psnr,i, (epoch-1)*len(data)+i)
                writer.add_scalar('Validation Loss',cur_loss,i, (epoch-1)*len(data)+i)
                writer.add_scalar('MSE Loss', mse_loss,i, (epoch-1)*len(data)+i)
        end=time.time()
        avg_loss/=len(data)
        avg_mse_loss/=len(data)
        all_ssim/=len(data)
        all_psnr/=len(data)
        timed=(end-start)/len(data)

        if (writer is not None): writer.add_text('Text','Test Avg. Loss: %f, MSE Loss: %f, Time: %f, PSNR: %.4f, SSIM %.4f' % (avg_loss, avg_mse_loss, timed, all_psnr, all_ssim))
        if (hyp_writer is not None):
            met_dict = {'hparam/SSIM':all_ssim,
                    'hparam/PSNR':all_psnr,
                    'hparam/Loss':avg_loss,
                    'hparam/MSELoss':avg_mse_loss}

            hyp_writer.add_hparams(self.hypers,met_dict,str(epoch))
        with open(config.data_path+'logs/'+self.args.alg+str(self.args.expir)+'.txt',"a") as f:
            f.write('Test Avg. Loss: %f, MSE Loss: %f, Time: %f, PSNR: %.4f, SSIM %.4f\n' % (avg_loss, avg_mse_loss, timed, all_psnr, all_ssim))
        print('Test Avg. Loss: %f, MSE Loss: %f, Time: %f, PSNR: %.4f, SSIM %.4f' % (avg_loss, avg_mse_loss, timed, all_psnr, all_ssim))

