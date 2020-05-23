import torch
from losses.sliced_sm import (
    sliced_score_matching,
    sliced_score_estimation,
    sliced_score_estimation_vr,
)
from losses.efficient_sm import esm_scorenet_VR
import numpy as np
import time


def wae_ssm(
    encoder,
    decoder,
    score,
    score_opt,
    X,
    training=True,
    n_energy_opt=1,
    n_particles=1,
    lam=10,
):
    start_time = time.time()
    z = encoder(X)
    ssm_loss, *_ = sliced_score_estimation_vr(score, z, n_particles=n_particles)
    if training:
        score_opt.zero_grad()
        ssm_loss.backward()
        score_opt.step()
        for i in range(n_energy_opt - 1):
            z = encoder(X)
            ssm_loss, *_ = sliced_score_estimation_vr(score, z, n_particles=n_particles)
            score_opt.zero_grad()
            ssm_loss.backward()
            score_opt.step()
    start_time -= time.time()

    z = encoder(X)
    decoded_X = decoder(z)
    recon = (X - decoded_X) ** 2
    recon = recon.sum(dim=(1, 2, 3))

    nlogpz = z ** 2 / 2.0 + np.log(2.0 * np.pi) / 2.0
    nlogpz = nlogpz.sum(dim=-1)

    scores = score(z)
    entropy_surrogate = (scores.detach() * z).sum(dim=-1)

    loss = recon + lam * (nlogpz + entropy_surrogate)

    loss = loss.mean()

    return loss, start_time, ssm_loss, recon


def wae_ssm_fd(
    encoder,
    decoder,
    score,
    score_opt,
    X,
    training=True,
    n_energy_opt=1,
    n_particles=1,
    lam=10,
):
    start_time = time.time()
    z = encoder(X)
    ssm_loss = esm_scorenet_VR(score, z)
    if training:
        score_opt.zero_grad()
        ssm_loss.backward()
        score_opt.step()
        for i in range(n_energy_opt - 1):
            z = encoder(X)
            ssm_loss = esm_scorenet_VR(score, z)
            score_opt.zero_grad()
            ssm_loss.backward()
            score_opt.step()
    start_time -= time.time()

    z = encoder(X)
    decoded_X = decoder(z)
    recon = (X - decoded_X) ** 2
    recon = recon.sum(dim=(1, 2, 3))

    nlogpz = z ** 2 / 2.0 + np.log(2.0 * np.pi) / 2.0
    nlogpz = nlogpz.sum(dim=-1)

    scores = score(z)
    entropy_surrogate = (scores.detach() * z).sum(dim=-1)

    loss = recon + lam * (nlogpz + entropy_surrogate)

    loss = loss.mean()

    return loss, start_time, ssm_loss, recon


def wae_kernel(encoder, decoder, estimator, X, lam=10):
    z = encoder(X)
    decoded_X = decoder(z)
    recon = (X - decoded_X) ** 2
    recon = recon.sum(dim=(1, 2, 3))

    nlogpz = z ** 2 / 2.0 + np.log(2.0 * np.pi) / 2.0
    nlogpz = nlogpz.sum(dim=-1)

    with torch.no_grad():
        scores = estimator.compute_gradients(z.unsqueeze(0))

    entropy_surrogate = (scores.squeeze(0).detach() * z).sum(dim=-1)

    loss = recon + lam * (nlogpz + entropy_surrogate)

    loss = loss.mean()
    return loss
