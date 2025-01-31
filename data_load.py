from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
import traceback
from IPython.display import display, clear_output
import config
from torch.autograd import Variable
import pydicom
from synthetic.spiral import DoubleSpiralDataset

##
## Custom class for loading data
##
class MyCustomDataset(Dataset):
    def __init__(self, percent, direc, transform,args):
        self.data_root = direc
        self.args=args
        self.transform = transform
        if (os.path.exists((os.path.join(self.data_root, '/labels')))):
            self.names = np.array([name for name in os.listdir((os.path.join(self.data_root, '/labels'+name)))])
        else: self.names = np.array(sorted([name for name in os.listdir(self.data_root)]))
        if percent<0:self.names = self.names[0:-1*percent]
        else: self.names = self.names[0:int(percent*len(self.names)//100)]
        self.count = len(self.names)


    def __getitem__(self, index):
        name = self.names[index]
        rayed = torch.Tensor()
        if (os.path.exists((os.path.join(self.data_root, '/labels')))):
            img = Image.open((os.path.join(self.data_root, '/images/'+name)))
            rayed = Image.open((os.path.join(self.data_root, '/labels/'+name)))
        else:
            if(self.args.setup==4):
                img = Image.fromarray(np.load((os.path.join(self.data_root, name))))
            elif (self.args.setup==5):
                img = Image.fromarray(load_dicom_as_array((os.path.join(self.data_root, name))))
            else:
                img =  Image.open((os.path.join(self.data_root, name)))
        img = self.transform(img)
        # img = (img-img.min())/(img.max()-img.min())
        return (rayed, img)

    def __len__(self):
        return self.count

def load_dicom_as_array(filepath):
    dicom_file = pydicom.dcmread(filepath)
    image_array = dicom_file.pixel_array.astype(np.float32)

    # print('data:', image_array.min(),image_array.max(), image_array.shape)
    
    # Normalize the image if needed (e.g., to [0, 1] range) TODO: very necessary
    image_array -= np.min(image_array)
    image_array /= np.max(image_array)
    
    return image_array
##
## Loading data for training/testing
##
def load_data(args,test=False):
    if not args.synthetic:
        data_train_loader, data_valid_loader, data_test_loader = [],[],[]
        if (not test):
            data_train = MyCustomDataset(args.dataperc,args.data_path+'train', transform=transforms.Compose([
                                                                                    transforms.Resize((config.size, config.size)),
                                                                                    transforms.ToTensor()
                                                                                    ]),args=args)

            data_train_loader = DataLoader(data_train, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)


            data_valid = MyCustomDataset(-args.valid,args.data_path+'valid',transform=transforms.Compose([
                                                                                    transforms.Resize((config.size, config.size)),
                                                                                    transforms.ToTensor()
                                                                                    ]),args=args)

            data_valid_loader = DataLoader(data_valid, batch_size=args.batch_size_valid, shuffle=False, num_workers=args.workers, drop_last=False)
        if (test):
            data_test = MyCustomDataset(-args.valid,args.data_path+'test', transform=transforms.Compose([
                                                                                    transforms.Resize((config.size, config.size)),
                                                                                    transforms.ToTensor()
                                                                                    ]),args=args)

            data_test_loader = DataLoader(data_test, batch_size=args.batch_size_valid, shuffle=False, num_workers=args.workers, drop_last=False)

        return (data_train_loader, data_valid_loader, data_test_loader)
    else:
        data_train_loader, data_valid_loader, data_test_loader = [],[],[]
        
        data_train_loader = DataLoader(DoubleSpiralDataset( ), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, drop_last=False)
        data_valid_loader = DataLoader(DoubleSpiralDataset(), batch_size=1000, shuffle=True, num_workers=args.workers, drop_last=False)
        data_test_loader = DataLoader(DoubleSpiralDataset(), batch_size=1000, shuffle=True, num_workers=args.workers, drop_last=False)
         

        return (data_train_loader, data_valid_loader, data_test_loader)

def load_data_present(args,test=False):
     
    data_test = MyCustomDataset(100,'/home/yasmin/projects/ct/data/present', transform=transforms.Compose([
                                                                            transforms.Resize((config.size, config.size)),
                                                                            transforms.ToTensor()
                                                                            ]),args=args)

    data_test_loader = DataLoader(data_test, batch_size=args.batch_size_valid, shuffle=False, num_workers=args.workers, drop_last=False)

    return (None, data_test_loader, data_test_loader)
     


##
## Creating noisy version/scans after Radon from the truth
##
def create(truths, mean):
    if(config.angles != 0):rayed = config.fwd_op_mod(truths)
    else: rayed=truths.clone()

    rayed += Variable(config.noise * mean * torch.randn(rayed.shape)).type_as(rayed)
    return rayed
