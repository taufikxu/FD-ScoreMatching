python train_dsm.py ./configs/default.yaml --loss_func mdsm_baseline --dataset mnist -gpu 0
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset mnist -gpu 1
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd_nop --dataset mnist -gpu 2

python train_ssm.py ./configs/default.yaml --loss_func ssm_fd --batch_size 64 --max_lr 0.00001 --dataset mnist -gpu 7
python train_ssm.py ./configs/default.yaml --loss_func ssm_fd_nop --batch_size 64 --max_lr 0.00001 --dataset mnist -gpu 7
python train_ssm.py ./configs/default.yaml --loss_func ssm --batch_size 64 --max_lr 0.00001 --dataset mnist -gpu 7
python train_ssm.py ./configs/default.yaml --loss_func ssm_vr --batch_size 64 --max_lr 0.00001 --dataset mnist -gpu 7


python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset mnist --batch_size 128 -gpu 4,5
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset fashionmnist --batch_size 128 -gpu 0,1
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset cifar --batch_size 128 -gpu 2,3
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset celeba32 --batch_size 128 -gpu 4,5

python sample_dsm.py ./configs/sample.yaml --old_model ./results/(train_dsm.py)_(mnist)_(2020-05-24-14-34-57)_((loss_func_mdsm_fd)(batch_size_128))_(None)/models/model300000.pt -gpu 0

python test_exact_sm.py ./configs/default.yaml -gpu 4,5 --old_model  ./results/\(train_ssm.py\)_\*/models/model300000.pt