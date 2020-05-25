import torch
import numpy as np
import Torture
from Utils import flags

FLAGS = flags.FLAGS

dim_dict = {"cifar": 32 * 32 * 3, "mnist": 28 * 28}


def exact_score_matching(energy_net, samples, dim):
    samples = samples.view(-1, dim)
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


def mdsm_baseline(energy, x_real, sigmas, sigma02, dim, mean=True):
    x_real = x_real.view(FLAGS.batch_size, dim)
    x_noisy = x_real + sigmas * torch.randn_like(x_real)

    x_noisy = x_noisy.requires_grad_()
    E = energy(x_noisy)
    grad_x = torch.autograd.grad(E.sum(), x_noisy, create_graph=True)[0]

    LS_loss = torch.sum(((x_real - x_noisy) / sigma02 + grad_x) ** 2, dim=-1)
    LS_loss = LS_loss / (sigmas.view(-1) ** FLAGS.dsm_pow)

    if mean is True:
        return LS_loss.mean()
    else:
        return LS_loss


def mdsm_tracetrick(energy, x_real, sigmas, sigma02, dim, mean=True):
    x_real = x_real.view(FLAGS.batch_size, dim)
    x_noisy = x_real + sigmas * torch.randn_like(x_real)
    projection = torch.randn_like(x_noisy).detach()

    x_noisy = x_noisy.requires_grad_()
    E = energy(x_noisy).sum()
    grad_x = torch.autograd.grad(E, x_noisy, create_graph=True)[0]

    LS_loss = torch.sum(((x_real - x_noisy) / sigma02 + grad_x) * projection, dim=-1)
    LS_loss = LS_loss ** 2
    # print(LS_loss.shape)
    LS_loss = LS_loss / (sigmas.view(-1) ** 2)
    if mean is True:
        return LS_loss.mean()
    else:
        return LS_loss


# def mdsm_ssm_fd(energy, x_real, sigmas, sigma02, dim):
#     batch_size = x_real.shape[0]
#     # print(esm_eps.shape)
#     x_real = x_real.view(batch_size, dim)
#     dsm_noise = sigmas * torch.randn_like(x_real)
#     x_noisy = x_real + dsm_noise
#     v_un = torch.randn_like(x_noisy)
#     v_norm = torch.sqrt(torch.sum(v_un ** 2, dim=-1, keepdim=True))
#     v = v_un / v_norm * np.sqrt(dim) * FLAGS.esm_eps

#     cat_input = torch.cat([x_noisy, x_noisy + v, x_noisy - v], 0)
#     cat_output = energy(cat_input)
#     # print(cat_output.shape)
#     out_1 = cat_output[:batch_size]
#     out_2 = cat_output[batch_size : 2 * batch_size]
#     out_3 = cat_output[2 * batch_size :]

#     diffs_1 = out_2 - out_3
#     loss1 = torch.mul(diffs_1, diffs_1) / 4
#     loss2 = (-out_2 - out_3 + 2 * out_1) * 2
#     loss_ssm = (loss1 + loss2).mean() / FLAGS.esm_eps ** 2

#     first_term = (-out_2 + out_3) * 0.5
#     second_term = torch.sum(v * (x_noisy - x_real), dim=-1) / sigma02
#     loss_dsm = (first_term + second_term) ** 2 / (sigmas.view(-1) ** FLAGS.dsm_pow)
#     loss_dsm = (loss_dsm / FLAGS.esm_eps ** 2).mean()

#     return loss_dsm + FLAGS.mixture_weight * loss_ssm


def mdsm_fd(energy, x_real, sigmas, sigma02, dim):
    batchSize = FLAGS.batch_size
    esm_eps = FLAGS.esm_eps
    # print(esm_eps.shape)
    x_real = x_real.view(batchSize, dim)
    dsm_noise = sigmas * torch.randn_like(x_real)
    x_noisy = x_real + dsm_noise
    v_un = torch.randn_like(x_noisy)
    v_norm = torch.sqrt(torch.sum(v_un ** 2, dim=-1, keepdim=True))
    v = v_un / v_norm * esm_eps

    cat_x = torch.cat([x_noisy + v, x_noisy - v], 0)
    logp = -energy(cat_x)

    logp1, logp2 = logp[:batchSize], logp[batchSize:]
    first_term = (logp1 - logp2) * 0.5
    second_term = torch.sum(v * (x_noisy - x_real), dim=-1) / sigma02
    LS_loss_e = (first_term + second_term) ** 2 / (sigmas.view(-1) ** 2)
    LS_loss_e = LS_loss_e / esm_eps ** 2 * dim
    # print(LS_loss_e.shape)
    LS_loss = (LS_loss_e).mean()
    return LS_loss


# def mdsm_fd_abs(energy, x_real, sigmas, sigma02, dim):
#     batchSize = FLAGS.batch_size
#     esm_eps = FLAGS.esm_eps
#     # print(esm_eps.shape)
#     x_real = x_real.view(batchSize, dim)
#     dsm_noise = sigmas * torch.randn_like(x_real)
#     # print(dsm_noise.shape)
#     x_noisy = x_real + dsm_noise
#     v_un = torch.randn_like(x_noisy)
#     v_norm = torch.sqrt(torch.sum(v_un ** 2, dim=-1, keepdim=True))
#     v = v_un / v_norm * esm_eps
#     # print(v.shape)

#     cat_x = torch.cat([x_noisy + v, x_noisy - v], 0)
#     logp = -energy(cat_x)

#     logp1, logp2 = logp[:batchSize], logp[batchSize:]
#     first_term = (logp1 - logp2) * 0.5
#     second_term = torch.sum(v * (x_noisy - x_real), dim=-1) / sigma02
#     if FLAGS.dsm_pow != 1:
#         sigmas_weight = sigmas.view(-1) ** FLAGS.dsm_pow
#     else:
#         sigmas_weight = sigmas.view(-1)
#     LS_loss_e = torch.abs(first_term + second_term) / sigmas_weight
#     LS_loss_e = LS_loss_e / esm_eps ** 2 * dim
#     # print(LS_loss_e.shape)
#     LS_loss = (LS_loss_e).mean()
#     return LS_loss


def mdsm_fd_nop(energy, x_real, sigmas, sigma02, dim):
    batchSize = FLAGS.batch_size
    x_real = x_real.view(batchSize, dim)
    dsm_noise = sigmas * torch.randn_like(x_real)
    x_noisy = x_real + dsm_noise
    v_un = torch.randn_like(x_noisy)
    v_norm = torch.sqrt(torch.sum(v_un ** 2, dim=-1, keepdim=True))
    v = v_un / v_norm * FLAGS.esm_eps

    logp1 = -energy(x_noisy + v)
    logp2 = -energy(x_noisy - v)

    first_term = (logp1 - logp2) * 0.5
    second_term = torch.sum(v * (x_noisy - x_real), dim=-1) / sigma02
    if FLAGS.dsm_pow != 1:
        sigmas_weight = sigmas.view(-1) ** FLAGS.dsm_pow
    else:
        sigmas_weight = sigmas.view(-1)
    LS_loss_e = (first_term + second_term) ** 2 / sigmas_weight
    LS_loss = LS_loss_e.mean() / FLAGS.esm_eps ** 2 * dim
    return LS_loss


def ssm(energy, x):
    dim = np.prod(x.shape[1:])
    x = x.view(FLAGS.batch_size, dim)
    x.requires_grad_(True)
    u = torch.randn_like(x).to(x.device)

    log_p = -energy(x)
    score = torch.autograd.grad(log_p.sum(), x, create_graph=True)[0]
    projection = (score * u).sum(dim=-1)
    # print(projection.shape)
    loss1 = (projection ** 2) * 0.5

    grad_mul_u = score * u
    hvp = torch.autograd.grad(grad_mul_u.sum(), x, create_graph=True)[0]
    loss2 = (hvp * u).sum(dim=-1)

    return (loss1 + loss2).mean()


def ssm_vr(energy, x, mean=True):
    dim = np.prod(x.shape[1:])
    x = x.view(FLAGS.batch_size, dim)
    x.requires_grad_(True)
    u = torch.randn_like(x).to(x.device)

    log_p = -energy(x)
    score = torch.autograd.grad(log_p.sum(), x, create_graph=True)[0]
    loss1 = (score ** 2).sum(dim=-1) * 0.5

    grad_mul_u = score * u
    hvp = torch.autograd.grad(grad_mul_u.sum(), x, create_graph=True)[0]
    loss2 = (hvp * u).sum(dim=-1)

    if mean is True:
        return (loss1 + loss2).mean()
    else:
        return loss1 + loss2


def ssm_fd(energy, samples):
    eps = FLAGS.esm_eps
    dim = np.prod(samples.shape[1:])
    samples = samples.view(FLAGS.batch_size, dim)

    vectors = torch.randn_like(samples)
    vectors = vectors / torch.norm(vectors, p=2, dim=-1, keepdim=True) * eps
    # print(samples.shape)

    batch_size = samples.shape[0]
    cat_input = torch.cat([samples, samples + vectors, samples - vectors], 0)
    cat_output = energy(cat_input)
    # print(cat_output.shape)
    out_1 = cat_output[:batch_size]
    out_2 = cat_output[batch_size : 2 * batch_size]
    out_3 = cat_output[2 * batch_size :]

    diffs_1 = out_2 - out_3
    loss1 = torch.mul(diffs_1, diffs_1) / 8
    loss2 = -out_2 - out_3 + 2 * out_1
    loss = (loss1 + loss2).mean() / FLAGS.esm_eps ** 2

    return loss


def ssm_fd_nop(energy, samples):
    eps = FLAGS.esm_eps
    dim = np.prod(samples.shape[1:])
    samples = samples.view(FLAGS.batch_size, dim)

    vectors = torch.randn_like(samples)
    vectors = vectors / torch.norm(vectors, p=2, dim=-1, keepdim=True) * eps
    # print(samples.shape)

    out_1 = energy(samples)
    out_2 = energy(samples + vectors)
    out_3 = energy(samples - vectors)

    diffs_1 = out_2 - out_3
    loss1 = torch.mul(diffs_1, diffs_1) / 8
    loss2 = -out_2 - out_3 + 2 * out_1
    loss = (loss1 + loss2).mean() / FLAGS.esm_eps ** 2

    return loss
