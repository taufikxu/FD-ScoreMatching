

######## 4gpu snapshot_fre 5000
# DSM: dsm
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --runner AnnealRunner --config anneal_dsm.yml --doc cifar10_anneal_DSM_bs128 | tee LOGS/CIFAR10/anneal_DSM_bs128.out

# SSM: ssm
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --runner AnnealRunner --config anneal_ssm.yml --doc cifar10_anneal_SSM-VR_bs128_4gpu > LOGS/CIFAR10/anneal_SSM-VR_bs128_4gpu.out

# ESM_scorenet: esm_scorenet
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --runner AnnealRunner --config anneal_fdssm.yml --doc cifar10_anneal_ESM-scorenet_bs128_4gpu > LOGS/CIFAR10/anneal_ESM-scorenet_bs128_4gpu.out

# ESM_scorenet_VR: esm_scorenet_VR
CUDA_VISIBLE_DEVICES=0,1,2,3 python -u main.py --runner AnnealRunner --config anneal_fdssmvr.yml --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu.out






# Sample DSM
CUDA_VISIBLE_DEVICES=5 python main.py --runner AnnealRunner --test -o samples/cifar10_anneal_DSM_bs128_4gpu --doc cifar10_anneal_DSM_bs128_4gpu

# Sample SSM
CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --runner AnnealRunner --test -o samples/cifar10_anneal_SSM-VR_bs128_4gpu_snap2000_164000 --doc cifar10_anneal_SSM-VR_bs128_4gpu_snap2000 > LOGS/CIFAR10/none2.out 2>&1 &

# Sample ESM_scorenet-VR
CUDA_VISIBLE_DEVICES=4 nohup python -u main.py --runner AnnealRunner --test -o samples/cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap2000_178000 --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap2000 > LOGS/CIFAR10/none1.out 2>&1 &

# Sample DSM pretrained
CUDA_VISIBLE_DEVICES=5 python main.py --runner AnnealRunner --test -o samples/cifar10 --doc cifar10






# FID DSM 4gpu
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_DSM_bs128_4gpu > LOGS/CIFAR10/anneal_DSM_bs128_4gpu_FID_epsoch180000.out 2>&1 &

# FID DSM
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_DSM_bs128 > LOGS/CIFAR10/anneal_DSM_bs128_FID_epoch170000.out 2>&1 &

# FID SSM
CUDA_VISIBLE_DEVICES=5 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_SSM-VR_bs128_4gpu > LOGS/CIFAR10/anneal_SSM-VR_bs128_4gpu_FID.out 2>&1 &

# FID ESM_scorenet
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_ESM-scorenet_bs128_4gpu > LOGS/CIFAR10/anneal_ESM-scorenet_bs128_4gpu_FID.out 2>&1 &

# FID ESM_scorenet_VR
CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu_FID_epoch165000.out 2>&1 &

# FID DSM pretrained
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10 > LOGS/CIFAR10/anneal_DSM_pretrained_FID.out 2>&1 &






# Save sample images DSM
CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --runner AnnealRunner --save_sampled_images -o samples/cifar10_anneal_DSM_bs128_4gpu --doc cifar10_anneal_DSM_bs128_4gpu > LOGS/CIFAR10/anneal_DSM_bs128_4gpu_SaveSampledImages.out 2>&1 &

# Save sample images SSM
CUDA_VISIBLE_DEVICES=5 nohup python -u main.py --runner AnnealRunner --save_sampled_images -o samples/cifar10_anneal_SSM-VR_bs128_4gpu --doc cifar10_anneal_SSM-VR_bs128_4gpu > LOGS/CIFAR10/anneal_SSM-VR_bs128_4gpu_SaveSampledImages.out 2>&1 &

# Save sample images ESM_scorenet
CUDA_VISIBLE_DEVICES=4 nohup python -u main.py --runner AnnealRunner --save_sampled_images -o samples/cifar10_anneal_ESM-VR-scorenet_bs128_4gpu --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu_SaveSampledImages.out 2>&1 &





######## 4gpu snapshot_fre 2000
# DSM
CUDA_VISIBLE_DEVICES=6 nohup python -u main.py --runner AnnealRunner --config anneal.yml --doc cifar10_anneal_DSM_bs128 > LOGS/CIFAR10/anneal_DSM_bs128.out 2>&1 &

# DSM 4gpu
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main.py --runner AnnealRunner --config anneal.yml --doc cifar10_anneal_DSM_bs128_4gpu_snap2000 > LOGS/CIFAR10/anneal_DSM_bs128_4gpu_snap2000.out 2>&1 &

# DSM 4gpu trace trick
CUDA_VISIBLE_DEVICES=4,1,2,3 nohup python -u main.py --runner AnnealRunner --config anneal.yml --doc cifar10_anneal_DSM-tracetrick_bs128_4gpu_snap2000 > LOGS/CIFAR10/anneal_DSM-tracetrick_bs128_4gpu_snap2000.out 2>&1 &

# SSM
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main.py --runner AnnealRunner --config anneal.yml --doc cifar10_anneal_SSM-VR_bs128_4gpu_snap2000 > LOGS/CIFAR10/anneal_SSM-VR_bs128_4gpu_snap2000.out 2>&1 &

# ESM_scorenet
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main.py --runner AnnealRunner --config anneal.yml --doc cifar10_anneal_ESM-scorenet_bs128_4gpu > LOGS/CIFAR10/anneal_ESM-scorenet_bs128_4gpu.out 2>&1 &

# ESM_scorenet_VR 128
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main.py --runner AnnealRunner --config anneal.yml --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap2000 > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu_snap2000.out 2>&1 &

# ESM_scorenet_VR 192
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main.py --runner AnnealRunner --config anneal.yml --doc cifar10_anneal_ESM-VR-scorenet_bs192_4gpu_snap2000 > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs192_4gpu_snap2000.out 2>&1 &











# FID DSM 4gpu
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_DSM_bs128_4gpu_snap2000 > LOGS/CIFAR10/anneal_DSM_bs128_4gpu_snap2000_FID.out 2>&1 &

# FID DSM 4gpu trace trick
CUDA_VISIBLE_DEVICES=4 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_DSM-tracetrick_bs128_4gpu_snap2000 > LOGS/CIFAR10/anneal_DSM-tracetrick_bs128_4gpu_snap2000_FID.out 2>&1 &

# FID SSM
CUDA_VISIBLE_DEVICES=3 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_SSM-VR_bs128_4gpu_snap2000 > LOGS/CIFAR10/anneal_SSM-VR_bs128_4gpu_snap2000_FID.out 2>&1 &

# FID ESM_scorenet_VR 128
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap2000 > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu_snap2000_FID.out 2>&1 &

# FID ESM_scorenet_VR 192
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_ESM-VR-scorenet_bs192_4gpu_snap2000 > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs192_4gpu_snap2000_FID.out 2>&1 &





# FID ESM_scorenet_VR 128 epoch 178000
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap2000 > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu_snap2000_FIDepoch178000.out 2>&1 &

# FID DSM 4gpu trace trick epoch 144000
CUDA_VISIBLE_DEVICES=4 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_DSM-tracetrick_bs128_4gpu_snap2000 > LOGS/CIFAR10/anneal_DSM-tracetrick_bs128_4gpu_snap2000_FIDepoch144000.out 2>&1 &

# FID DSM 4gpu epoch 184000
CUDA_VISIBLE_DEVICES=0 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_DSM_bs128_4gpu_snap2000 > LOGS/CIFAR10/anneal_DSM_bs128_4gpu_snap2000_FIDepoch184000.out 2>&1 &



######## 4gpu snapshot_fre 1000

# SSM snap1000 seed 2020
CUDA_VISIBLE_DEVICES=0,2,6,7 nohup python -u main.py --runner AnnealRunner --config anneal.yml --doc cifar10_anneal_SSM-VR_bs128_4gpu_snap1000 > LOGS/CIFAR10/anneal_SSM-VR_bs128_4gpu_snap1000.out 2>&1 &

# SSM snap1000 seed 2021
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main.py --runner AnnealRunner --config anneal.yml --doc cifar10_anneal_SSM-VR_bs128_4gpu_snap1000_2 > LOGS/CIFAR10/anneal_SSM-VR_bs128_4gpu_snap1000_2.out 2>&1 &

# ESM_scorenet_VR snap1000 128 seed 2020
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main.py --runner AnnealRunner --config anneal.yml --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap1000 > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu_snap1000.out 2>&1 &

# ESM_scorenet_VR snap1000 128 seed 2021
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main.py --runner AnnealRunner --config anneal.yml --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_2 > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_2.out 2>&1 &





# SSM snap1000 seed 2020
CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python -u main.py --runner AnnealRunner --config anneal.yml --doc celebA_anneal_SSM-VR_bs128_4gpu_snap1000 > LOGS/celebA/anneal_SSM-VR_bs128_4gpu_snap1000.out 2>&1 &

# ESM_scorenet_VR snap1000 128 seed 2020
CUDA_VISIBLE_DEVICES=0,1,2,3 nohup python -u main.py --runner AnnealRunner --config anneal.yml --doc celebA_anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_2 > LOGS/celebA/anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_2.out 2>&1 &






# FID ESM_scorenet_VR
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap1000 > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_FID.out 2>&1 &

# FID ESM_scorenet_VR 2
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_2 > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_2_FID.out 2>&1 &

# FID SSM
CUDA_VISIBLE_DEVICES=5 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_SSM-VR_bs128_4gpu_snap1000 > LOGS/CIFAR10/anneal_SSM-VR_bs128_4gpu_snap1000_FID.out 2>&1 &

# FID SSM 2
CUDA_VISIBLE_DEVICES=6 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_SSM-VR_bs128_4gpu_snap1000_2 > LOGS/CIFAR10/anneal_SSM-VR_bs128_4gpu_snap1000_2_FID.out 2>&1 &




# Save sample images SSM 2
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --runner AnnealRunner --test -o samples/cifar10_anneal_SSM-VR_bs128_4gpu_snap1000_2_152000 --doc cifar10_anneal_SSM-VR_bs128_4gpu_snap1000_2 > LOGS/CIFAR10/anneal_SSM-VR__bs128_4gpu_snap1000_2_152000_SaveSampledImages.out 2>&1 &

# Save sample images ESM_scorenet
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --runner AnnealRunner --test -o samples/cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap1000 --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap1000 > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_SaveSampledImages.out 2>&1 &

# Save sample images ESM_scorenet 2
CUDA_VISIBLE_DEVICES=2 nohup python -u main.py --runner AnnealRunner --test -o samples/cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_2_199000 --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_2 > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_2_199000_SaveSampledImages.out 2>&1 &





# FID ESM_scorenet_VR 50000 samples
CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap1000 > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_FID_194000.out 2>&1 &

# FID ESM_scorenet_VR_2 50000 samples
CUDA_VISIBLE_DEVICES=6 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_2 > LOGS/CIFAR10/anneal_ESM-VR-scorenet_bs128_4gpu_snap1000_FID_149000.out 2>&1 &

# FID SSM_2 50000 samples
CUDA_VISIBLE_DEVICES=5 nohup python -u main.py --runner AnnealRunner --calculate_fid --doc cifar10_anneal_SSM-VR_bs128_4gpu_snap1000_2 > LOGS/CIFAR10/anneal_SSM-VR_bs128_4gpu_snap1000_2_FID_152000.out 2>&1 &


