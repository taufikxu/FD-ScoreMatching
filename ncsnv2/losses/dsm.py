import torch
import torch.autograd as autograd
import numpy as np


def anneal_dsm_score_estimation(
    scorenet, samples, sigmas, labels=None, anneal_power=2.0, hook=None
):
    if labels is None:
        labels = torch.randint(
            0, len(sigmas), (samples.shape[0],), device=samples.device
        )
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise
    target = -1 / (used_sigmas ** 2) * noise
    scores = scorenet(perturbed_samples, labels)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = (
        1
        / 2.0
        * ((scores - target) ** 2).sum(dim=-1)
        * used_sigmas.squeeze() ** anneal_power
    )

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)


def anneal_sliced_score_estimation_vr(
    scorenet, samples, sigmas, labels=None, anneal_power=2.0, hook=None, n_particles=1
):
    if labels is None:
        labels = torch.randint(
            0, len(sigmas), (samples.shape[0],), device=samples.device
        )
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    dup_samples = (
        perturbed_samples.unsqueeze(0)
        .expand(n_particles, *samples.shape)
        .contiguous()
        .view(-1, *samples.shape[1:])
    )
    dup_labels = (
        labels.unsqueeze(0).expand(n_particles, *labels.shape).contiguous().view(-1)
    )
    dup_samples.requires_grad_(True)

    # use Rademacher
    vectors = torch.randn_like(dup_samples)

    dup_samples.requires_grad_()
    grad1 = scorenet(dup_samples, dup_labels)
    gradv = torch.sum(grad1 * vectors)
    grad2 = autograd.grad(gradv, dup_samples, create_graph=True)[0]

    grad1 = grad1.view(dup_samples.shape[0], -1)
    loss1 = torch.sum(grad1 * grad1, dim=-1) / 2.0

    loss2 = torch.sum((vectors * grad2).view(dup_samples.shape[0], -1), dim=-1)

    loss1 = loss1.view(n_particles, -1).mean(dim=0)
    loss2 = loss2.view(n_particles, -1).mean(dim=0)

    loss = (loss1 + loss2) * (used_sigmas.squeeze() ** anneal_power)

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)


def anneal_ESM_scorenet_VR(
    scorenet,
    samples,
    sigmas,
    labels=None,
    anneal_power=2.0,
    hook=None,
    n_particles=1,
    eps=0.1,
):
    data_dim = np.prod(samples.shape[1:])
    if labels is None:
        labels = torch.randint(
            0, len(sigmas), (samples.shape[0],), device=samples.device
        )
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    dup_samples = (
        perturbed_samples.unsqueeze(0)
        .expand(n_particles, *samples.shape)
        .contiguous()
        .view(-1, *samples.shape[1:])
    )
    dup_labels = (
        labels.unsqueeze(0).expand(n_particles, *labels.shape).contiguous().view(-1)
    )
    dup_samples.requires_grad_(True)

    # use Rademacher
    vectors = torch.randn_like(dup_samples)
    vectors = (
        vectors / torch.sqrt(torch.sum(vectors ** 2, dim=[1, 2, 3], keepdim=True)) * eps
    )
    # vectors = vectors / torch.sqrt(torch.sum(vectors**2, dim=[1,2,3], keepdim=True)) * 10. * used_sigmas

    cat_input = torch.cat([dup_samples + vectors, dup_samples - vectors], 0)
    cat_label = torch.cat([dup_labels, dup_labels], 0)

    grad_all = scorenet(cat_input, cat_label)

    batch_size = dup_samples.shape[0]
    grad_all = grad_all.view(2 * batch_size, -1)
    vectors = vectors.view(batch_size, -1)
    out_1 = grad_all[:batch_size]
    out_2 = grad_all[batch_size:]

    grad1 = out_1 + out_2
    grad2 = out_1 - out_2

    loss_1 = (grad1 * grad1) / 8.0
    loss_2 = grad2 * vectors * (data_dim / (2.0 * eps * eps))  # for CIFAR-10
    loss = torch.sum(loss_1 + loss_2, dim=-1)

    loss = loss * (used_sigmas.squeeze() ** anneal_power)

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)
