import argparse




def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

# Training settings
parser = argparse.ArgumentParser(description='CT comparison framework')
parser.add_argument('--batch-size', type=int, default=20, metavar='N',
                    help='input batch size (default: 20)')
parser.add_argument('--epochs', type=int, default=10, metavar='NE',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--noise', type=float, default=3.2, metavar='N',
                    help='noise level of the mean for the images (default: 0.01), i.e. 1 percent')
parser.add_argument('--seed', type=int, default=10, metavar='SD',
                    help='random seed (default: 10)')
parser.add_argument('--setup', type=int, default=1, metavar='S',
                    help='setup (different for different datasets, see config)(default: 1)')
parser.add_argument('--log_interval', type=int, default=2, metavar='L',
                    help='how many batches to wait before logging training status')
parser.add_argument('--iterates', type=int, default=10,
                    help='how many iterates for algorithms like LG or LPD/ For iterative schemes it is the number of steps for gradient descent')
parser.add_argument('--dataperc', type=float, default=100,
                    help='how much data should be used, default: 100%')
parser.add_argument('--cp_interval', type=int, default=5000, metavar='C',
                    help='how many batches to wait before saving a checkpoint')
parser.add_argument('--data-path', type=str, default='/local/scratch/public/zs334/compFrm/',#default='/home/zs334/rds/hpc-work/CompFrm/',
                    help='path to the data')
parser.add_argument('--alg', type=str, default='compare',
                    help='FBP,TV,FBP+U,FL,LG,LPD,TV,CLAR')
parser.add_argument('--cuda', type=str2bool, default=True,
                    help='use CUDA for training?')
parser.add_argument('--outp', type=str2bool, default=True,
                    help='output intermideate results?')
parser.add_argument('--visual', type=str2bool, default=False,
                    help='Just visualise and not train')
parser.add_argument('--init', type=str2bool, default=True,
                    help='Initialise weights differntly?')
parser.add_argument('--load', type=str, default='', metavar='LD',
                    help='Load path?')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--detectors', default=700, type=int, metavar='DET',
                    help='number of detectors for Ray Transform')
parser.add_argument('--mu', type=float, default=10, metavar='MU',
                    help='Value for constant mu')
parser.add_argument('--clamping', type=float, default=0, metavar='CLAMP',
                    help='Clamping value')
parser.add_argument('--eps', type=float, default=1e-5, metavar='EPS',
                    help='Value for constant eps')
parser.add_argument('--expir', type=int, default=-1, metavar='EXP',
                    help='Experiment number to run with')
parser.add_argument('--clip', type=float, default=1, metavar='CLP',
                    help='Clip value for the gradient')
parser.add_argument('--valid', type=int, default=100, metavar='EXP',
                    help='Number of images to validate on')
parser.add_argument('--size', type=int, default=16, metavar='EXP',
                    help='Size as a parameter')
parser.add_argument('--mult', type=str2bool, default=False, metavar='MLT',
                    help='Multiple instances running? So different checkpoint loaders')
parser.add_argument('--wclip', type=str2bool, default=True, metavar='WCL',
                    help='Clip weights to 0?')
# add an argument test_only
parser.add_argument('--test', type=str2bool, default=False, metavar='TST',
                    help='Test only')
# add an argument setting=sparse or limited
parser.add_argument('--setting', type=str, default='limited', metavar='SET',
                    help='sparse or limited')
# add an argument alpha
parser.add_argument('--alpha', type=float, default=0.0, metavar='ALP',
                    help='alpha value for TV regularization')
# add an argument test_mode
parser.add_argument('--test_mode', type=str, default='GD', metavar='TM',
                    help='test mode')
# add an argument K
parser.add_argument('--K', type=int, default=1, metavar='K',
                    help='K value for inner loop')  

# add an argument n_channels
parser.add_argument('--n_channels', type=int, default=32, metavar='NC',
                    help='number of channels')

# add an argument n_layers
parser.add_argument('--n_layers', type=int, default=10, metavar='NL',
                    help='number of layers')

# add an argument alpha_psm
parser.add_argument('--alpha_psm', type=float, default=1e-5, metavar='ALP_psm',
                    help='alpha value for PSM')

# add a comment
parser.add_argument('--comment', type=str, default='', metavar='CMT', help='comment')

# add an argument synthetic
parser.add_argument('--synthetic', type=str2bool, default=False, metavar='SYN', help='synthetic data')

# add an argument momentum
parser.add_argument('--momentum', type=float, default=0.5, metavar='MOM', help='momentum')

# add an argument alpha_inv
parser.add_argument('--alpha_inv', type=float, default=1e-5, metavar='ALP_inv', help='alpha value for inverse problem')

# add an argument batch_size_valid
parser.add_argument('--batch_size_valid', type=int, default=1, metavar='BSV', help='batch size for validation')