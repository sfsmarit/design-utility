import numpy as np


def fs(pitch, vf, k11, k12):
    return vf / pitch * (1 - (k11+k12) / 2 / np.pi)


def fp(pitch, vf, k11, k12, k2):
    return vf / pitch * (1 - (k11+k12) / 2 / np.pi + k2 * 4 / np.pi**2)


def fr(pitch, vf, k11, k12):
    return fs(pitch, vf, k11, -k12)


def k2_eff(fs, fp):
    """Return effective coupling k^2 derived from x/tan(x)."""
    x = np.pi / 2 * fs / fp
    return x / np.tan(x)


def k2_bvd(fs, fp):
    """Return BVD-defined electromechanical coupling k_t^2 = 1 - (fs/fp)^2."""
    return 1 - (fs/fp)**2


def fs_gradient(pitch, vf, k11, k12, vf_grad, k11_grad, k12_grad):
    """Calculate the gradient of fs with respect to vf, k11, and k12."""
    df_dvf = 1 / pitch * (1 - (k11 + k12) / 2 / np.pi)
    df_dk11 = -vf / pitch / 2 / np.pi
    df_dk12 = -vf / pitch / 2 / np.pi
    return df_dvf * vf_grad + df_dk11 * k11_grad + df_dk12 * k12_grad


def fp_gradient(pitch, vf, k11, k12, k2, vf_grad, k11_grad, k12_grad, k2_grad):
    """Calculate the gradient of fp with respect to vf, k11, k12, and k2."""
    df_dvf = 1 / pitch * (1 - (k11 + k12) / 2 / np.pi + k2 * 4 / np.pi**2)
    df_dk11 = -vf / pitch / 2 / np.pi
    df_dk12 = -vf / pitch / 2 / np.pi
    df_dk2 = vf / pitch * 4 / np.pi**2
    return df_dvf * vf_grad + df_dk11 * k11_grad + df_dk12 * k12_grad + df_dk2 * k2_grad


def fr_gradient(pitch, vf, k11, k12, vf_grad, k11_grad, k12_grad):
    """Calculate the gradient of fr with respect to vf, k11, and k12."""
    return fs_gradient(pitch, vf, k11, -k12, vf_grad, k11_grad, k12_grad)
