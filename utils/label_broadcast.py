import torch
import torch.nn as nn


def label_broadcast(label_map,target):
    # label_map is the prediction output through softmax operation
    N, C, W, H = label_map.shape
    # label_map = label_map.softmax(dim=1)
    new_label = label_map.clone()
    mask = (target.unsqueeze(1) != 255).detach()
    new_mask = torch.zeros((N, 1, W, H)).cuda()
    left = label_map[:, :, 0:W - 1, :] * mask[:, :, 0:W - 1, :]
    right = label_map[:, :, 1:W, :] * mask[:, :, 1:W, :]
    up = label_map[:, :, :, 0:H - 1] * mask[:, :, :, 0:H - 1]
    down = label_map[:, :, :, 1:H] * mask[:, :, :, 1:H]
    new_label[:, :, 1:W, :] = new_label[:, :, 1:W, :].clone() + left
    new_label[:, :, 0:W - 1] = new_label[:, :, 0:W - 1].clone() + right
    new_label[:, :, :, 1:H] = new_label[:, :, :, 1:H].clone() + down
    new_label[:, :, :, 0:H - 1] = new_label[:, :, :, 0:H - 1].clone() + up
    new_label = nn.Softmax(dim=1)(new_label)

    new_mask[:, :, 1:W, :] += mask[:, :, 0:W - 1, :]
    new_mask[:, :, 0:W - 1] += mask[:, :, 1:W, :]
    new_mask[:, :, :,1:H] += mask[:, :, :, 0:H - 1]
    new_mask[:, :, :, 0:H-1] += mask[:, :, :, 1:H]
    new_mask = new_mask>=1
    return new_label,new_mask.squeeze().detach()
