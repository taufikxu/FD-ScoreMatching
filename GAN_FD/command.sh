python training_gan.py ./configs/fd-gan.yaml -subfolder 11.12 -trainer.kwargs.reg_type real_fake  


python training_gan.py ./configs/fd-gan.yaml -subfolder 11.12 -trainer.kwargs.reg_type real  -training.fd_eps 0.01
python training_gan.py ./configs/fd-gan.yaml -subfolder 11.12 -trainer.kwargs.reg_type real  -training.fd_eps 0.001
python training_gan.py ./configs/fd-gan.yaml -subfolder 11.12 -trainer.kwargs.reg_type real  -training.fd_eps 0.0001
python training_gan.py ./configs/fd-gan.yaml -subfolder 11.12 -trainer.kwargs.reg_type real  -training.fd_eps 0.00001