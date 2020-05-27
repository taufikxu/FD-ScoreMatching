The following are the commands to reproduce our quantitative results in this paper. GPU_ID can be an integer (such as 0) or multiple integers separated by commas (such as 0,1).

## Training command

For baseline:
```
# EBM with MNIST on DSM
python train_dsm.py ./configs/default.yaml --loss_func mdsm_baseline --dataset mnist -gpu GPU_ID

# EBM with MNIST on SSM/SSM_VR
python train_ssm.py ./configs/default.yaml --loss_func ssm --batch_size 64 --max_lr 0.00001 --dataset mnist -gpu GPU_ID
python train_ssm.py ./configs/default.yaml --loss_func ssm_vr --batch_size 64 --max_lr 0.00001 --dataset mnist -gpu GPU_ID
```

The FD-version of DSM:
```
# On MNIST and FD(non-parallel):
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset mnist -gpu GPU_ID
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd_nop --dataset mnist -gpu GPU_ID

# On FashionMNIST and Celeba32
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset fashionmnist --batch_size 128 -gpu GPU_ID
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset celeba32 --batch_size 128 -gpu GPU_ID
```


The FD-version of SSM:
```
# On MNIST and FD(non-parallel):
python train_ssm.py ./configs/default.yaml --loss_func ssm_fd --batch_size 64 --max_lr 0.00001 --dataset mnist -gpu GPU_ID
python train_ssm.py ./configs/default.yaml --loss_func ssm_fd_nop --batch_size 64 --max_lr 0.00001 --dataset mnist -gpu GPU_ID
```

## Test Command:
In the following, the MODEL_PATH is the path for the checkpoint file, i.e., the pth file that contains the trained model parameters.

The command for image generation is as follows:
```
python sample_dsm.py ./configs/sample.yaml --old_model MODEL_PATH --gpu GPU_ID
```

The command for testing exact SM loss:
```
python test_exact_sm.py ./configs/default.yaml --old_model MODEL_PATH --gpu GPU_ID
```