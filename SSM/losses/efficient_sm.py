import torch

# import torch.autograd as autograd
# import numpy as np


def single_efficient_score_matching(energy_net, samples, eps=0.01, noise_type="sphere"):
    samples.requires_grad_(True)
    vectors = torch.randn_like(samples)
    if noise_type == "radermacher":
        vectors = vectors.sign() * eps
    elif noise_type == "sphere":
        vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True) * eps

    batch_size = samples.shape[0]
    cat_input = torch.cat([samples, samples + vectors], 0)
    cat_output = energy_net(cat_input)
    out_1 = cat_output[:batch_size]
    out_2 = cat_output[batch_size:]

    diffs = out_1 - out_2
    loss = torch.sum(torch.mul(diffs, diffs) + 4 * diffs)

    return loss


def dsm_fd(energy_net, samples, sigma=1, eps=0.001):
    batchSize, dim = samples.shape[0], samples.shape[1]
    samples.requires_grad_(True)
    vector = torch.randn_like(samples) * sigma
    perturbed_inputs = samples + vector

    noise = torch.randn_like(samples)
    noise_norm = torch.sqrt(torch.sum(noise ** 2, dim=-1, keepdim=True))
    noise = noise / noise_norm * eps * (dim ** 0.5)

    cat_input = torch.cat([perturbed_inputs + noise, perturbed_inputs - noise])

    logp = -energy_net(cat_input)
    dlogpv = sigma ** 2 * (logp[:batchSize] - logp[batchSize:]) * 0.5
    kernel = torch.sum(vector * noise, dim=-1)
    loss = (dlogpv + kernel) ** 2
    loss = loss.mean() / (2.0 * eps ** 2)
    return loss


def efficient_score_matching_conjugate(
    energy_net, samples, noise=None, eps=0.1, noise_type="sphere", detach=False
):
    dim = samples.shape[1]
    # samples.requires_grad_(True)
    if noise is None:
        noise = torch.randn_like(samples)
        noise_norm = torch.sqrt(torch.sum(noise ** 2, dim=-1, keepdim=True))
        noise = noise / noise_norm * eps
    vectors = noise

    batch_size = samples.shape[0]
    cat_input = torch.cat([samples, samples + vectors, samples - vectors], 0)
    cat_output = energy_net(cat_input)
    out_1 = cat_output[:batch_size]
    out_2 = cat_output[batch_size : 2 * batch_size]
    out_3 = cat_output[2 * batch_size :]

    diffs_1 = out_2 - out_3
    loss1 = (diffs_1 ** 2) / 8
    loss2 = -out_2 - out_3 + 2 * out_1
    if detach is True:
        loss1 = loss1.detach()
        loss2 = loss2.detach()
    loss = (loss1 + loss2).mean() / (eps ** 2) * dim

    return loss


def esm_scorenet_VR(scorenet, samples, eps=0.1):
    # use Rademacher
    dim = samples.shape[-1]
    vectors = torch.randn_like(samples)
    vectors = vectors / torch.sqrt(torch.sum(vectors ** 2, dim=1, keepdim=True)) * eps

    cat_input = torch.cat([samples + vectors, samples - vectors], 0)

    grad_all = scorenet(cat_input)
    batch_size = samples.shape[0]
    grad_all = grad_all.view(2 * batch_size, -1)
    vectors = vectors.view(batch_size, -1)
    out_1 = grad_all[:batch_size]
    out_2 = grad_all[batch_size:]

    grad1 = out_1 + out_2
    grad2 = out_1 - out_2

    loss_1 = torch.sum((grad1 * grad1) / 8.0, dim=-1)
    loss_2 = torch.sum(grad2 * vectors * (dim / (2 * eps * eps)), dim=-1)
    loss = (loss_1 + loss_2).mean(dim=0)

    return loss


def MLE_efficient_score_matching_conjugate(
    energy_net, samples, eps=0.01, mle_ratio=10, noise_type="sphere"
):
    samples.requires_grad_(True)
    vectors = torch.randn_like(samples)
    if noise_type == "radermacher":
        vectors = vectors.sign() * eps
    elif noise_type == "sphere":
        vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True) * eps

    batch_size = samples.shape[0]
    cat_input = torch.cat([samples, samples + vectors, samples - vectors], 0)
    cat_output = energy_net(cat_input)
    out_1 = cat_output[:batch_size]
    out_2 = cat_output[batch_size : 2 * batch_size]
    out_3 = cat_output[2 * batch_size :]

    diffs_1 = out_2 - out_3
    loss = torch.sum(
        (torch.mul(diffs_1, diffs_1) / 8) - out_2 - out_3 + mle_ratio * out_1
    )

    return loss
