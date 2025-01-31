# Learning Difference-of-Convex Regularizers for Inverse Problems: A Flexible Framework with Theoretical Guarantees


## Environment Installation
```bash
conda env create -f environment.yml
```
If the installation fails using the environment.yml file, you can install the required packages manually.
## Prepare Data(Mayo Grand Challenge)
Please go to the [link](https://aapm.app.box.com/s/eaw4jddb53keg1bptavvvd1sf4x3pe9h) and download the data. Please put the data in the following path:
```bash
./data/[test/valid/train]
```

In all of our experiments, we use 1mm B30 data of the highest precision. 



## Run Our Algorithm(ADCR)
example call:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --setup=5 --dataperc=100 --epochs=1 --lr=1e-4 --eps=1e-6 --alg=ADCR --iterates=200 --valid=128 --test=True --batch-size=10 --noise=3.2 --seed=10  --setting=limited --load=./data/nets_new/ADCR/limited/limited.pt --test_mode=GD 
```
- epochs: number of epochs
- lr: learning rate (used to train the neural network)
- eps: step size for the optimization 
- alg: algorithm to be used (ADR, TV, FBP, ADCR etc.), please take a look at files names in the Algorithms folder
- iterates: maximum number of iterations for the optimization
- valid: number of validation images
- batch-size: batch size
- detectors: number of detectors
- noise: noise level
- load: path to the checkpoint
- seed: random seed
- setting: limited or sparse
- test_mode: GD or CCP or PSM
- K: number of innner loop iterations for the CCP/PSM algorithm

## Train a regularizer
example call:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --setup=5 --dataperc=100 --epochs=20 --lr=1e-4 --eps=1e-5 --alg=ADCR --iterates=100 --valid=10 --batch-size=10  --gpu=1 --noise=3.2 --load=False --seed=10 --wclip=True --setting=sparse --mu=10
```


## Credits
This code is based on the [CT_framework](https://github.com/Zakobian/CT_framework_).



## Description of other methods
* FBP (Filtered back-projection) 
* TV (Total Variation)
* ADR (Adversarial Regularizer): https://arxiv.org/abs/1805.11572
* LG (Learned gradient descent): https://arxiv.org/abs/1704.04058
* LPD (Learned primal dual): https://arxiv.org/abs/1707.06474
* FL (Fully learned): https://nature.com/articles/nature25988.pdf
* FBP+U (FBP with a U-Net denoiser): https://arxiv.org/abs/1505.04597
* ACR: https://arxiv.org/abs/2008.02839
* ACNCR: https://openreview.net/forum?id=yavtWi6ew9
* AWCR: https://arxiv.org/abs/2402.01052
* ADCR(ours)

In order to add your own algorithms to the list, create a new file in the **Algorithms** folder in the form *name*.py and use BaseAlg.py as the template.

