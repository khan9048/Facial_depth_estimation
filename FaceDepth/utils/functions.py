import torch
from torch.autograd import Variable
import torch.nn as nn
import numpy as np


def get_coords(b, h, w):
    i_range = Variable(torch.arange(0, h).view(1, h, 1).expand(b, 1, h, w))  # [B, 1, H, W]
    j_range = Variable(torch.arange(0, w).view(1, 1, w).expand(b, 1, h, w))  # [B, 1, H, W]
    coords = torch.cat((j_range, i_range), dim=1)
    norm = Variable(torch.Tensor([w, h]).view(1, 2, 1, 1))
    coords = coords * 2. / norm - 1.
    coords = coords.permute(0, 2, 3, 1)

    return coords


def resize_tensor(img, coords):
    return nn.functional.grid_sample(img, coords, mode='bilinear', padding_mode='zeros')


def imgrad(img):
    img = torch.mean(img, 1, True)
    fx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    conv1 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fx).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv1.weight = nn.Parameter(weight)
    grad_x = conv1(img)

    fy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    conv2 = nn.Conv2d(1, 1, kernel_size=3, stride=1, padding=1, bias=False)
    weight = torch.from_numpy(fy).float().unsqueeze(0).unsqueeze(0)
    if img.is_cuda:
        weight = weight.cuda()
    conv2.weight = nn.Parameter(weight)
    grad_y = conv2(img)

    #     grad = torch.sqrt(torch.pow(grad_x,2) + torch.pow(grad_y,2))

    return grad_y, grad_x


def imgrad_yx(img):
    N, C, _, _ = img.size()
    grad_y, grad_x = imgrad(img)
    return torch.cat((grad_y.view(N, C, -1), grad_x.view(N, C, -1)), dim=1)


def reg_scalor(grad_yx):
    return torch.exp(-torch.abs(grad_yx) / 255.)
