The following are the commands to reproduce our quantitative results in this paper. Our code is largely based on [the original code of NCSN](https://github.com/ermongroup/ncsn).

## NCSN 
The running commands for NCSN are as follows:

For training:

```
# SSMVR
python main.py --runner AnnealRunner --config anneal_ssm.yml --doc cifar10_anneal_SSMVR_bs128_4gpu

# FD-SSMVR
python main.py --runner AnnealRunner --config anneal_fdssmvr.yml --doc cifar10_anneal_FD-SSMVR_bs128_4gpu
```

For generating samples (grid):
```
# SSMVR
python main.py --runner AnnealRunner --test -o samples/cifar10_anneal_SSMVR_bs128_4gpu --doc cifar10_anneal_SSMVR_bs128_4gpu

# FD-SSMVR
python main.py --runner AnnealRunner --test -o samples/cifar10_anneal_FD-SSMVR_bs128_4gpu --doc cifar10_anneal_FD-SSMVR_bs128_4gpu
```

For calculate FID:
```
# SSMVR
python main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_SSMVR_bs128_4gpu

# FD-SSMVR
python main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_FD-SSMVR_bs128_4gpu
```

For saving sampled images:
```
# SSMVR
python main.py --runner AnnealRunner --save_sampled_images -o samples/cifar10_anneal_SSMVR_bs128_4gpu --doc cifar10_anneal_SSMVR_bs128_4gpu

# FD-SSMVR
python main.py --runner AnnealRunner --save_sampled_images -o samples/cifar10_anneal_FD-SSMVR_bs128_4gpu --doc cifar10_anneal_FD-SSMVR_bs128_4gpu

```
