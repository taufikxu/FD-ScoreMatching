python train_dsm.py ./configs/default.yaml --loss_func mdsm_baseline --dataset mnist --batch_size 64 --subfolder Final_nsp --activation nsoftplus -gpu 0
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset mnist --batch_size 64 --subfolder Final_nsp --activation nsoftplus -gpu 1
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd_nop --dataset mnist --batch_size 64 --subfolder Final_nsp --activation nsoftplus -gpu 2

python train_ssm.py ./configs/default.yaml --loss_func ssm_fd --batch_size 64 --max_lr 0.00001 --dataset mnist --subfolder Final_nsp --activation nsoftplus -gpu 3
python train_ssm.py ./configs/default.yaml --loss_func ssm_fd_nop --batch_size 64 --max_lr 0.00001 --dataset mnist --subfolder Final_nsp --activation nsoftplus -gpu 4
python train_ssm.py ./configs/default.yaml --loss_func ssm --batch_size 64 --max_lr 0.00001 --dataset mnist --subfolder Final_nsp --activation nsoftplus -gpu 5
python train_ssm.py ./configs/default.yaml --loss_func ssm_vr --batch_size 64 --max_lr 0.00001 --dataset mnist --subfolder Final_nsp --activation nsoftplus -gpu 6

# For OOD
python train_dsm.py ./configs/default.yaml --subfolder FinalOODnsp --loss_func mdsm_baseline --dataset svhn --activation nsoftplus -gpu 0,1
python train_dsm.py ./configs/default.yaml --subfolder FinalOODnsp  --loss_func mdsm_fd --dataset svhn --activation nsoftplus -gpu 2,3
python train_dsm.py ./configs/default.yaml --subfolder FinalOODnsp --loss_func mdsm_baseline --dataset cifar --activation nsoftplus -gpu 4,5
python train_dsm.py ./configs/default.yaml --subfolder FinalOODnsp  --loss_func mdsm_fd --dataset cifar --activation nsoftplus -gpu 6,7
python train_dsm.py ./configs/default.yaml --subfolder FinalOODnsp --loss_func mdsm_baseline --dataset imagenet --activation nsoftplus -gpu 0,1,2,3,4,5,6,7
python train_dsm.py ./configs/default.yaml --subfolder FinalOODnsp  --loss_func mdsm_fd --dataset imagenet --activation nsoftplus -gpu 0,1,2,3,4,5,6,7

python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset mnist --subfolder Final_sample_nsp --activation nsoftplus -gpu 0,1
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset fashionmnist --subfolder Final_sample_nsp --activation nsoftplus -gpu 2,3
python train_dsm.py ./configs/default.yaml --loss_func mdsm_fd --dataset celeba32 --subfolder Final_sample_nsp --activation nsoftplus -gpu 4,5


python sample_dsm.py ./configs/sample.yaml --old_model ./results/OOD/(train_dsm.py)_(svhn)_(2020-05-28-21-16-13)_((loss_func_mdsm_fd))_(None)/models/model300000.pt -gpu 0
python test_exact_sm.py ./configs/default.yaml -gpu 4,5,6,7 --old_model  './results/Final1/(train_dsm.py)_(mnist)_(2020-05-30-21-28-28)_((loss_func_mdsm_baseline)(batch_size_64))_(None)/models/model200000.pt'
python test_exact_sm.py ./configs/default.yaml -gpu 4,5,6,7 --old_model  '/home/kunxu/Workspace/EBM_based/ESM_EBM/results/0AllOld/(ESM_EBM)(train_dsm.py_mnist)_(2020-05-02-21-50-17)_(loss_func_mdsm_baseline)_(None)/models/model300000.pt'

python test_ood.py ./configs/default.yaml -old_model './results/OOD/(train_dsm.py)_(cifar)_(2020-05-29-21-33-54)_((loss_func_mdsm_fd))_(None)/models/model300000.pt' -gpu 0