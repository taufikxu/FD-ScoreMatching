import torch
import numpy as np
from Utils import flags

FLAGS = flags.FLAGS


def score_matching(energy_net, samples):
    samples = samples.view(-1, 784)
    samples.requires_grad_(True)
    logp = -energy_net(samples).sum()
    grad1 = torch.autograd.grad(logp, samples)[0]
    loss1 = (torch.norm(grad1, dim=-1) ** 2 / 2.0).detach()

    loss2 = torch.zeros(samples.shape[0], device=samples.device)
    for i in range(samples.shape[1]):
        logp = -energy_net(samples).sum()
        grad1 = torch.autograd.grad(logp, samples, create_graph=True)[0]
        grad = torch.autograd.grad(grad1[:, i].sum(), samples)[0][:, i]
        loss2 += grad.detach()

    loss = loss1 + loss2
    return loss.mean()


def dsm_multi_scale(energy, x_real, sigmas, sigma02):
    x_noisy = x_real + sigmas * torch.randn_like(x_real)

    x_noisy = x_noisy.requires_grad_()
    E = energy(x_noisy).sum()
    grad_x = torch.autograd.grad(E, x_noisy, create_graph=True)[0]

    LS_loss = torch.sum(((x_real - x_noisy) / sigma02 + grad_x) ** 2, dim=(1, 2, 3))
    LS_loss = (LS_loss / (sigmas.view(-1) ** 2)).mean()
    return LS_loss


def dsm_multi_scale_fd(energy, x_real, sigmas, sigma02):
    batchSize = FLAGS.batch_size
    dsm_noise = sigmas * torch.randn_like(x_real)
    x_noisy = x_real + dsm_noise
    v_un = torch.randn_like(x_noisy)
    v_norm = torch.sqrt(torch.sum(v_un ** 2, dim=(1, 2, 3), keepdim=True))
    v = v_un / v_norm * np.sqrt(784) * FLAGS.esm_eps

    cat_x = torch.cat([x_noisy + v, x_noisy - v], 0)
    logp = -energy(cat_x)

    logp1, logp2 = logp[:batchSize], logp[batchSize:]
    first_term = (logp1 - logp2) * 0.5
    second_term = torch.sum(v * (x_noisy - x_real), dim=(1, 2, 3)) / sigma02
    LS_loss_e = (first_term + second_term) ** 2 / sigmas.view(-1) ** 2
    LS_loss = LS_loss_e.mean() / FLAGS.esm_eps ** 2
    return LS_loss


def dsm_multi_scale_fd_nop(energy, x_real, sigmas, sigma02):
    dsm_noise = sigmas * torch.randn_like(x_real)
    x_noisy = x_real + dsm_noise
    v_un = torch.randn_like(x_noisy)
    v_norm = torch.sqrt(torch.sum(v_un ** 2, dim=(1, 2, 3), keepdim=True))
    v = v_un / v_norm * np.sqrt(784) * FLAGS.esm_eps

    # cat_x = torch.cat([x_noisy + v, x_noisy - v], 0)
    # logp = -energy(cat_x)
    # logp1, logp2 = logp[:batchSize], logp[batchSize:]
    logp1 = -energy(x_noisy + v)
    logp2 = -energy(x_noisy - v)

    first_term = (logp1 - logp2) * 0.5
    second_term = torch.sum(v * (x_noisy - x_real), dim=(1, 2, 3)) / sigma02
    LS_loss_e = (first_term + second_term) ** 2 / sigmas.view(-1) ** 2
    LS_loss = LS_loss_e.mean() / FLAGS.esm_eps ** 2
    return LS_loss


def ssm(energy, x):
    x.requires_grad_(True)

    u = torch.randn_like(x).to(x.device)

    log_p = -energy(x)
    score = torch.autograd.grad(log_p.sum(), x, create_graph=True)[0]

    loss1 = (score * u).sum(dim=-1) * 0.5

    grad_mul_u = score * u
    hvp = torch.autograd.grad(grad_mul_u.sum(), x, create_graph=True)[0]
    loss2 = (hvp * u).sum(dim=-1)

    return (loss1 + loss2).mean()


def ssm_vr(energy, x):
    x.requires_grad_(True)

    u = torch.randn_like(x).to(x.device)

    log_p = -energy(x)
    score = torch.autograd.grad(log_p.sum(), x, create_graph=True)[0]

    loss1 = (score ** 2).sum(dim=-1) * 0.5

    grad_mul_u = score * u
    hvp = torch.autograd.grad(grad_mul_u.sum(), x, create_graph=True)[0]
    loss2 = (hvp * u).sum(dim=-1)

    return (loss1 + loss2).mean()


def ssm_fd(energy, samples):
    eps = FLAGS.esm_eps
    vectors = torch.randn_like(samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True) * eps
    # print(samples.shape)

    batch_size = samples.shape[0]
    cat_input = torch.cat([samples, samples + vectors, samples - vectors], 0)
    cat_output = energy(cat_input)
    out_1 = cat_output[:batch_size]
    out_2 = cat_output[batch_size : 2 * batch_size]
    out_3 = cat_output[2 * batch_size :]

    diffs_1 = out_2 - out_3
    loss1 = torch.mul(diffs_1, diffs_1) / 4
    loss2 = (-out_2 - out_3 + 2 * out_1) * 2
    loss = (loss1 + loss2).mean() / FLAGS.esm_eps ** 2

    return loss


def ssm_fd_nop(energy, samples):
    eps = FLAGS.esm_eps
    vectors = torch.randn_like(samples)
    vectors = vectors / torch.norm(vectors, dim=-1, keepdim=True) * eps
    # print(samples.shape)

    # batch_size = samples.shape[0]
    # cat_input = torch.cat([samples, samples + vectors, samples - vectors], 0)
    # cat_output = energy(cat_input)
    # out_1 = cat_output[:batch_size]
    # out_2 = cat_output[batch_size:2 * batch_size]
    # out_3 = cat_output[2 * batch_size:]
    out_1 = energy(samples)
    out_2 = energy(samples + vectors)
    out_3 = energy(samples - vectors)

    diffs_1 = out_2 - out_3
    loss1 = torch.mul(diffs_1, diffs_1) / 4
    loss2 = (-out_2 - out_3 + 2 * out_1) * 2
    loss = (loss1 + loss2).mean() / FLAGS.esm_eps ** 2

    return loss


loss_dict = {
    "dsm_multi_scale": dsm_multi_scale,
    "dsm_multi_scale_fd": dsm_multi_scale_fd,
    "dsm_multi_scale_fd_nop": dsm_multi_scale_fd_nop,
    "ssm": ssm,
    "ssm_vr": ssm_vr,
    "ssm_fd": ssm_fd,
    "ssm_fd_nop": ssm_fd_nop,
}
