from collections import OrderedDict
from .sync_batchnorm import convert_model
import torch
from .DeeplabV2 import *


def freeze_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = False


def release_bn(net):
    for module in net.modules():
        if isinstance(module, torch.nn.modules.BatchNorm2d):
            for i in module.parameters():
                i.requires_grad = True


def init_model(cfg):
    _, _,model_tar = Deeplab(num_classes=cfg.num_classes, init_stu=cfg.init_weight)
    # model_stu = convert_model(model_stu)
    # model_stu = nn.DataParallel(model_stu, device_ids=[0, 1])
    # model_tea_src = convert_model(model_tea_src)
    # model_tea_src = nn.DataParallel(model_tea_src, device_ids=[0, 1])
    # # model_tea_tar = convert_model(model_tea_tar)
    # model_tea_tar = nn.DataParallel(model_tea_tar, device_ids=[0,1])

    # model = nn.DataParallel(model,device_ids=[0])

    # if cfg.fixbn:
    #     freeze_bn(model)
    # else:
    #     release_bn(model)

    # checkpoint = torch.load(cfg.init_weight)
    #
    # if 'state_dict' in checkpoint:
    #     model.load_state_dict(checkpoint['state_dict'])
    # else:
    #     model.module.load_state_dict(checkpoint)

    # self.logger.info("Checkpoint loaded successfully from " + filename)
    #     if cfg.model=='deeplab' and cfg.init_weight != 'None':
    #         params = torch.load(cfg.init_weight)
    #         print('Model restored with weights from : {}'.format(cfg.init_weight))
    #         if 'init-' in cfg.init_weight and cfg.model=='deeplab':
    #             new_params = model.state_dict().copy()
    #             for i in params:
    #                 i_parts = i.split('.')
    #                 if not i_parts[1] == 'layer5':
    #                     new_params['.'.join(i_parts[1:])] = params[i]
    #             model.load_state_dict(new_params, strict=True)
    #
    #         else:
    #             new_params = model.state_dict().copy()
    #             for i in params:
    #                 if 'module' in i:
    #                     i_ = i.replace('module.', '')
    #                     new_params[i_] = params[i]
    #                 elif i in ['epoch','iteration','state_dict','optimizer','best_MIou']:
    #                     pass
    #                 else:
    #                     new_params[i] = params[i]
    # #                i_parts = i.split('.')[0]
    #             model.load_state_dict(new_params, strict=True)
    # #            model.load_state_dict(params, strict=True)
    #
    #     if cfg.restore_from != 'None':
    #         params = torch.load(cfg.restore_from)
    #         model.load_state_dict(params)
    #         print('Model initialize with weights from : {}'.format(cfg.restore_from))

    # if cfg.multigpu:
    #     model = convert_model(model)
    #     model = nn.DataParallel(model,device_ids=[0,1])
    # if cfg.train:
    #     model.train().cuda()
    #     print('Mode --> Train')
    # else:
    #     model.eval().cuda()
    #     print('Mode --> Eval')
    model_tar.train().cuda()
    # discriminator.train().cuda()
    # model_tea_tar.eval().cuda()
    # model_tea_src.eval().cuda()
    # print("zzzzzzzzzzzzzzzzzz")
    return model_tar#,model_tea_src,model_tea_tar
