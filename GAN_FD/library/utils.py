import torch
from torch import distributions
from Tools import FLAGS
from library.models import inception_score


def get_zdist(dtype, dim):
    # Get distribution
    device = FLAGS.device
    if dtype == "uniform":
        low = -torch.ones(dim, device=device)
        high = torch.ones(dim, device=device)
        zdist = distributions.Uniform(low, high)
    elif dtype == "gauss":
        mu = torch.zeros(dim, device=device)
        scale = torch.ones(dim, device=device)
        zdist = distributions.Normal(mu, scale)
    else:
        raise NotImplementedError

    # Add dim attribute
    zdist.dim = dim

    return zdist


def get_ydist(dtype, dim):
    device = FLAGS.device
    logits = torch.zeros(dim, device=device)
    ydist = distributions.categorical.Categorical(logits=logits)

    # Add nlabels attribute
    ydist.nlabels = dim

    return ydist


def update_average(model_tgt, model_src, beta=0.999):
    param_dict_src = dict(model_src.named_parameters())
    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert p_src is not p_tgt
        p_tgt.data.mul_(beta).add_((1 - beta) * p_src.data)


class Evaluator(object):
    def __init__(
        self, generator, zdist, ydist, batch_size=64, inception_nsamples=50000,
    ):
        self.generator = generator
        self.zdist = zdist
        self.ydist = ydist
        self.inception_nsamples = inception_nsamples
        self.batch_size = batch_size

    def compute_inception_score(self):
        self.generator.eval()
        imgs = []
        while len(imgs) < self.inception_nsamples:
            ztest = self.zdist.sample((self.batch_size,))
            ytest = self.ydist.sample((self.batch_size,))

            samples = self.generator(ztest, ytest)
            samples = [s.data.cpu().numpy() for s in samples]
            imgs.extend(samples)

        imgs = imgs[: self.inception_nsamples]
        score, score_std = inception_score.inception_score(imgs, resize=True, splits=10)

        return score, score_std

    def create_samples(self):
        self.generator.eval()
        z = self.zdist.sample((self.batch_size,))
        y = self.ydist.sample((self.batch_size,))
        with torch.no_grad():
            x = self.generator(z, y)
        return x
