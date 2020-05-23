python train_dsm.py ./configs/default.yaml --loss_func mdsm_baseline --batch_size 64 --dataset mnist -gpu 0 
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --batch_size 64 --dataset mnist -gpu 1
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd_nop --batch_size 64 --dataset mnist -gpu 2

python train_ssm.py ./configs/default.yaml --loss_func ssm_fd --batch_size 64 --dataset mnist -gpu 3
python train_ssm.py ./configs/default.yaml --loss_func ssm_fd_nop --batch_size 64 --dataset mnist -gpu 4
python train_ssm.py ./configs/default.yaml --loss_func ssm --batch_size 64 --dataset mnist -gpu 5
python train_ssm.py ./configs/default.yaml --loss_func ssm_vr --batch_size 64 --dataset mnist -gpu 6

python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset mnist --batch_size 128 -gpu 0,1
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset fashionmnist --batch_size 128 -gpu 2,3
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset celeba32 --batch_size 128 -gpu 4,5
