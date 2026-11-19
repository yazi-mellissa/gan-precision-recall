"""GAN losses. Each D loss returns (loss, stats) and each G loss returns (loss, stats).

Kept:  bce_d / bce_g (non-saturating GAN loss), optional R1 penalty.
Tried: hinge loss, PR-divergence (chi-squared critic, Verine et al. 2023), PR-divergence on a BCE critic.
"""
import torch
import torch.nn.functional as F


def r1_penalty(out_real, x_real, gamma):
    """R1 (Mescheder et al. 2018): gamma/2 * ||grad_x D(x)||^2 on real images. x_real must require grad."""
    grad = torch.autograd.grad(outputs=out_real.sum(), inputs=x_real, create_graph=True)[0]
    return (gamma / 2.0) * grad.pow(2).sum(dim=1).mean()


# ----------------------------------------------------------------------------- kept

def bce_d(D, x_real, x_fake, r1_gamma=0.0):
    """BCE discriminator loss on logits (numerically stable), with optional R1 penalty."""
    if r1_gamma > 0.0:
        x_real.requires_grad_(True)
    logits_real = D(x_real, logits=True)
    logits_fake = D(x_fake, logits=True)
    loss = F.binary_cross_entropy_with_logits(logits_real, torch.ones_like(logits_real)) + \
           F.binary_cross_entropy_with_logits(logits_fake, torch.zeros_like(logits_fake))
    r1 = r1_penalty(logits_real, x_real, r1_gamma) if r1_gamma > 0.0 else torch.zeros(())
    stats = {"D_real": torch.sigmoid(logits_real).mean().item(),
             "D_fake": torch.sigmoid(logits_fake).mean().item(), "r1": r1.item()}
    return loss + r1, stats


def bce_g(D, x_fake):
    """Non-saturating generator loss: -log D(G(z))."""
    logits_fake = D(x_fake, logits=True)
    return F.binary_cross_entropy_with_logits(logits_fake, torch.ones_like(logits_fake)), {}


# ----------------------------------------------------------------------------- tried, not kept

def hinge_d(D, x_real, x_fake, r1_gamma=0.0):
    """Hinge discriminator loss (Lim & Ye 2017; Miyato et al. 2018), with optional R1 penalty."""
    if r1_gamma > 0.0:
        x_real.requires_grad_(True)
    out_real = D(x_real, logits=True)
    out_fake = D(x_fake, logits=True)
    loss = F.relu(1.0 - out_real).mean() + F.relu(1.0 + out_fake).mean()
    r1 = r1_penalty(out_real, x_real, r1_gamma) if r1_gamma > 0.0 else torch.zeros(())
    return loss + r1, {"D_real": out_real.mean().item(), "D_fake": out_fake.mean().item(), "r1": r1.item()}


def hinge_g(D, x_fake):
    return -D(x_fake, logits=True).mean(), {}


def ratio_from_chi2(T):
    """Density ratio p/p_G from the chi-squared critic: r = max(T/2 + 1, 0)."""
    return torch.clamp(T / 2 + 1, min=0)


def chi2_d(D, x_real, x_fake, r1_gamma=0.0, t_clip=None):
    """Chi-squared critic loss for the PR-divergence, with optional R1 penalty.

    Without a Lipschitz control the critic T is unbounded and diverges (E4/E6: T_real > 1e8).
    t_clip bounds T before g*(t) = t^2/4 + t, so one sample cannot dominate the loss.
    """
    if r1_gamma > 0.0:
        x_real.requires_grad_(True)
    T_real = D(x_real, logits=True)
    T_fake = D(x_fake.detach(), logits=True)
    T_fake_g = T_fake.clamp(-2.0, t_clip) if t_clip else T_fake
    loss = -(T_real.mean() - (T_fake_g ** 2 / 4 + T_fake_g).mean())
    r1 = r1_penalty(T_real, x_real, r1_gamma) if r1_gamma > 0.0 else torch.zeros(())
    stats = {"T_real": T_real.mean().item(), "T_fake": T_fake.mean().item(),
             "r_real": ratio_from_chi2(T_real).mean().item(), "r_fake": ratio_from_chi2(T_fake).mean().item(),
             "r1": r1.item()}
    return loss + r1, stats


def prdiv_g(D, x_fake, lam=1.0, t_clip=None, soft_tau=0.0):
    """PR-divergence generator loss E[f_lam(r(G(z)))], with f_lam(r) = max(lam*r, 1) - max(lam, 1).

    The max gives no gradient to samples with lam*r < 1; for lam < 1 that is almost every sample
    (logged as share_nonzero_grad), which is why this loss could not train G (E9).
    soft_tau > 0 replaces the max by tau*logsumexp(lam*r/tau, 1/tau) to keep a small gradient.
    """
    T = D(x_fake, logits=True)
    if t_clip:
        T = T.clamp(-2.0, t_clip)
    r = ratio_from_chi2(T)
    if soft_tau > 0.0:
        stacked = torch.stack([lam * r / soft_tau, torch.full_like(r, 1.0 / soft_tau)])
        f = soft_tau * torch.logsumexp(stacked, dim=0) - max(lam, 1.0)
    else:
        f = torch.maximum(lam * r, torch.ones_like(r)) - max(lam, 1.0)
    return f.mean(), {"share_nonzero_grad": (lam * r > 1).float().mean().item(),
                      "share_r_clamped": (r <= 0).float().mean().item(), "mean_r_fake": r.mean().item()}


def prdiv_g_bce(D, x_fake, lam=1.0):
    """PR-divergence generator loss with the ratio read from a BCE discriminator: r = exp(logit) = D/(1-D)."""
    r = torch.exp(D(x_fake, logits=True)).clamp(max=100.0)  # clamp against extreme logits
    f = torch.maximum(lam * r, torch.ones_like(r)) - max(lam, 1.0)
    return f.mean(), {"share_nonzero_grad": (lam * r > 1).float().mean().item(),
                      "share_r_clamped": 0.0, "mean_r_fake": r.mean().item()}
