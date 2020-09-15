python training_gan.py ./configs/baseline.yaml -key baseline
python training_gan.py ./configs/baseline.yaml -trainer.kwargs.reg_type none -key baseline.noreg

python training_gan.py ./configs/fd-gan.yaml -key fd-reg
python training_gan.py ./configs/fd-gan.yaml -trainer.kwargs.reg_type none -key fd.noreg