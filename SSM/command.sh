python main.py --runner DKEFRunner --config dkef/dkef_parkinsons.yml --doc dkef_parkinsons_ssm_vr
python main.py --runner DKEFRunner --config dkef/dkef_parkinsons_single.yml --doc dkef_parkinsons_ssm_single
python main.py --runner DKEFRunner --config dkef/dkef_parkinsons_fd.yml --doc dkef_parkinsons_ssm_fd


python main.py --runner DKEFRunner --config dkef/dkef_redwine.yml --doc dkef_redwine_ssm_vr
python main.py --runner DKEFRunner --config dkef/dkef_redwine_single.yml --doc dkef_redwine_ssm_single
python main.py --runner DKEFRunner --config dkef/dkef_redwine_fd.yml --doc dkef_redwine_ssm_fd

python main.py --runner DKEFRunner --config dkef/dkef_whitewine.yml --doc dkef_whitewine_ssm_vr
python main.py --runner DKEFRunner --config dkef/dkef_whitewine_single.yml --doc dkef_whitewine_ssm_single
python main.py --runner DKEFRunner --config dkef/dkef_whitewine_fd.yml --doc dkef_whitewine_ssm_fd


python main.py --runner VAERunner --config vae/mnist_ssm.yml --doc vae_mnist_ssm --test
python main.py --runner VAERunner --config vae/mnist_ssm_fd.yml --doc vae_mnist_ssm_fd --test
python main.py --runner VAERunner --config vae/celeba_ssm.yml --doc vae_celeba_ssm --test 
python main.py --runner VAERunner --config vae/celeba_ssm_fd.yml --doc vae_celeba_ssm_fd --test


python main.py --runner NICERunner --config nice/nice_efficient_sm_conjugate.yml --ESM_eps 0.1 --doc nice_mnist_esm
python main.py --runner NICERunner --config nice/nice_ssm.yml --ESM_eps 0.1 --doc nice_mnist_ssm
python main.py --runner NICERunner --config nice/nice_ssm_vr.yml --ESM_eps 0.1 --doc nice_mnist_ssm_vr

python main.py --runner WAERunner --config wae/mnist_ssm.yml --doc wae_mnist_ssm --test
python main.py --runner WAERunner --config wae/mnist_ssm_fd.yml --doc wae_mnist_ssm_fd --test
python main.py --runner WAERunner --config wae/celeba_ssm.yml --doc wae_celeba_ssm --test
python main.py --runner WAERunner --config wae/celeba_ssm_fd.yml --doc wae_celeba_ssm_fd --test