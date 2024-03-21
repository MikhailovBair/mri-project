import math
import numpy as np
import pylab as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

gyromagnetic_ratio = 42.58 # MHz / T ~ kHz / mT

def getRotMat(B: torch.Tensor):
    """
    Args:
        B: field of shape [3 x N]
    """
    R = torch.zeros(3, 3, B.shape[-1], device=B.device)
    R[0][1] = B[2]
    R[0][2] = -B[1]
    R[1][0] = -B[2]
    R[1][2] = B[0]
    R[2][0] = B[1]
    R[2][1] = -B[0]
    return R.permute(2, 0, 1)


def getR(B: torch.Tensor, dt: float):
    """
    Args:
        B: field matrix of shape [3 x N]
        dt: timestep
    """
    Bnorm = B.norm(dim=0)
    nx = B[0] / Bnorm
    ny = B[1] / Bnorm
    nz = B[2] / Bnorm
    phi = dt * gyromagnetic_ratio * Bnorm * 2 * math.pi
    R = torch.zeros(B.shape[1], 3, 3, device=B.device)
    R[:, 0, 0] = nx ** 2 + (1 - nx ** 2) * torch.cos(phi)
    R[:, 0, 1] = nx * ny * (1 - torch.cos(phi)) + nz * torch.sin(phi)
    R[:, 0, 2] = nx * nz * (1 - torch.cos(phi)) - ny * torch.sin(phi)
    R[:, 1, 0] = nx * ny * (1 - torch.cos(phi)) - nz * torch.sin(phi)
    R[:, 1, 1] = nx ** 2 + (1 - nx ** 2) * torch.cos(phi)
    R[:, 1, 2] = ny * nz * (1 - torch.cos(phi)) + nx * torch.sin(phi)
    R[:, 2, 0] = nx * nz * (1 - torch.cos(phi)) + ny * torch.sin(phi)
    R[:, 2, 1] = ny * nz * (1 - torch.cos(phi)) - nx * torch.sin(phi)
    R[:, 2, 2] = nz ** 2 + (1 - nz ** 2) * torch.cos(phi)
    return R