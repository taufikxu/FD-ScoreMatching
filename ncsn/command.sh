######## Training ########
# SSMVR
python main.py --runner AnnealRunner --config anneal_ssm.yml --doc cifar10_anneal_SSMVR_bs128_4gpu

# FD-SSMVR
python main.py --runner AnnealRunner --config anneal_fdssmvr.yml --doc cifar10_anneal_FD-SSMVR_bs128_4gpu



######## Generate samples (grid) ########
# SSMVR
python main.py --runner AnnealRunner --test -o samples/cifar10_anneal_SSMVR_bs128_4gpu --doc cifar10_anneal_SSMVR_bs128_4gpu

# FD-SSMVR
python main.py --runner AnnealRunner --test -o samples/cifar10_anneal_FD-SSMVR_bs128_4gpu --doc cifar10_anneal_FD-SSMVR_bs128_4gpu



######## Calculate FID ########
# SSMVR
python main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_SSMVR_bs128_4gpu

# FD-SSMVR
python main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_FD-SSMVR_bs128_4gpu



######## Save sampled images ########
# SSMVR
python main.py --runner AnnealRunner --save_sampled_images -o samples/cifar10_anneal_SSMVR_bs128_4gpu --doc cifar10_anneal_SSMVR_bs128_4gpu

# FD-SSMVR
python main.py --runner AnnealRunner --save_sampled_images -o samples/cifar10_anneal_FD-SSMVR_bs128_4gpu --doc cifar10_anneal_FD-SSMVR_bs128_4gpu
