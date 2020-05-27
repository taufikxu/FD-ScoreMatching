The following are the commands to reproduce our quantitative results in this paper. GPU_ID can be an integer (such as 0) or multiple integers separated by commas (such as 0,1).

## DKEF

The training command for parkinsons dataset with SSM-VR/SSM/FD-SSM objectives.
```
python main.py --runner DKEFRunner --config dkef/dkef_parkinsons.yml --doc dkef_parkinsons_ssm_vr
python main.py --runner DKEFRunner --config dkef/dkef_parkinsons_single.yml --doc dkef_parkinsons_ssm_single
python main.py --runner DKEFRunner --config dkef/dkef_parkinsons_fd.yml --doc dkef_parkinsons_ssm_fd
```

The training command for RedWine dataset with SSM-VR/SSM/FD-SSM objectives.
```
python main.py --runner DKEFRunner --config dkef/dkef_redwine.yml --doc dkef_redwine_ssm_vr
python main.py --runner DKEFRunner --config dkef/dkef_redwine_single.yml --doc dkef_redwine_ssm_single
python main.py --runner DKEFRunner --config dkef/dkef_redwine_fd.yml --doc dkef_redwine_ssm_fd
```

The training command for WhiteWine dataset with SSM-VR/SSM/FD-SSM objectives.
```
python main.py --runner DKEFRunner --config dkef/dkef_whitewine.yml --doc dkef_whitewine_ssm_vr
python main.py --runner DKEFRunner --config dkef/dkef_whitewine_single.yml --doc dkef_whitewine_ssm_single
python main.py --runner DKEFRunner --config dkef/dkef_whitewine_fd.yml --doc dkef_whitewine_ssm_fd
```

The models will be also evaluated on the test data after training. 

## NICE 
The training code for NICE is as follows:

```
# MNIST with Approx BP
python main.py --runner NICERunner --config nice/nice_kingma.yml --doc nice_mnist_kingma

# MNIST with CP
python main.py --runner NICERunner --config nice/nice_S.yml --doc nice_mnist_CP

# MNIST with SSM objectives
python main.py --runner NICERunner --config nice/nice_ssm.yml --doc nice_mnist_ssm

# MNIST with SSM-VR objectives
python main.py --runner NICERunner --config nice/nice_ssm_vr.yml --doc nice_mnist_ssm_vr

# MNIST with FD-SSM objectives
python main.py --runner NICERunner --config nice/nice_efficient_sm_conjugate.yml --ESM_eps 0.1 --doc nice_mnist_esm

# MNIST with dsm(0.1)
python main.py --runner NICERunner --config nice/nice_dsm.yml --dsm_sigma 0.1 --doc nice_mnist_dsm0.1

# MNIST with dsm(1.74)
python main.py --runner NICERunner --config nice/nice_dsm.yml --dsm_sigma 1.74 --doc nice_mnist_dsm1.74
```

## VAE/WAE with implicit encoders

The training code for WAE is as follows:
```
# MNIST with SSM objectives
python main.py --runner VAERunner --config vae/mnist_ssm.yml --doc vae_mnist_ssm
# MNIST with FD-SSM objectives
python main.py --runner VAERunner --config vae/mnist_ssm_fd.yml --doc vae_mnist_ssm_fd

# CelebA with SSM objectives
python main.py --runner VAERunner --config vae/celeba_ssm.yml --doc vae_celeba_ssm
# CelebA with FD-SSM objectives
python main.py --runner VAERunner --config vae/celeba_ssm_fd.yml --doc vae_celeba_ssm_fd
```

The training code for WAE is as follows:
```
# MNIST with SSM objectives
python main.py --runner WAERunner --config wae/mnist_ssm.yml --doc wae_mnist_ssm
# MNIST with FD-SSM objectives
python main.py --runner WAERunner --config wae/mnist_ssm_fd.yml --doc wae_mnist_ssm_fd

# CelebA with SSM objectives
python main.py --runner WAERunner --config wae/celeba_ssm.yml --doc wae_celeba_ssm
# CelebA with FD-SSM objectives
python main.py --runner WAERunner --config wae/celeba_ssm_fd.yml --doc wae_celeba_ssm_fd
```

To test the performance, the argument '--test' should be appeneded to the above commands. For example, the test command for VAE on MNIST with ssm objectives is as follows:
```
python main.py --runner VAERunner --config vae/mnist_ssm_fd.yml --doc vae_mnist_ssm_fd --test
```
 