import torch
import torch.nn.functional as F
from scipy.spatial.distance import cdist
import numpy as np
import torch.nn as nn
def obtain_label(outputs,net,class_num):
    start_test = True
    with torch.no_grad():
        # inputs = inputs.cuda()
        # N,C,H,W=outputs.shape()
        # outputs=net(inputs)
        feas = net.get_features()
        feas=F.interpolate(feas, (600,600), mode='bilinear', align_corners=True)
        outputs = outputs.permute(0, 2, 3, 1).contiguous().view(-1, class_num)
        feas = feas.permute(0, 2, 3, 1).contiguous().view(-1, class_num)
        all_fea = feas.float().cpu()
        all_output = outputs.float().cpu()
        # all_label = labels.float()
        # start_test = False

    all_output = nn.Softmax(dim=1)(all_output)
    ent = torch.sum(-all_output * torch.log(all_output + 1e-5), dim=1)
    unknown_weight = 1 - ent / np.log(class_num)
    _, predict = torch.max(all_output, 1)

    # accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    # if args.distance == 'cosine':
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()

    all_fea = all_fea.float().cpu().numpy()
    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    cls_count = np.eye(K)[predict].sum(axis=0)
    labelset = np.where(cls_count > 0)
    labelset = labelset[0]
    # print(labelset)

    dd = cdist(all_fea, initc[labelset], 'cosine')
    pred_label = dd.argmin(axis=1)
    pred_label = labelset[pred_label]

    for round in range(1):
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc[labelset], 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = labelset[pred_label]

    # acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)
    # log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)

    # args.out_file.write(log_str + '\n')
    # args.out_file.flush()
    # print(log_str + '\n')

    return pred_label.astype('int')