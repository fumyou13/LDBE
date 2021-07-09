import torch
from utils.optimize import adjust_learning_rate
from .base_trainer import BaseTrainer
from utils.flatwhite import *
from easydict import EasyDict as edict
import os.path as osp
from dataset import dataset
import neptune
import math
from PIL import Image
from utils.meters import AverageMeter, GroupAverageMeter
import torchvision.transforms.functional as tf
import torch.nn.functional as F
import operator
import pickle
import random
import copy
from utils.kmeans import kmeans_cluster
from utils.func import Acc, thres_cb_plabel, gene_plabel_prop, mask_fusion
from utils.pool import Pool
from utils.flatwhite import *
from trainer.base_trainer import *
from utils.shot import *
import time


criterion_nll = nn.NLLLoss()





def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-7
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy

class EMA(object):
    def __init__(self, model, alpha):
        self.step = 0
        self.model = copy.deepcopy(model)
        self.alpha = alpha

    def update(self, model):
        decay = min(1 - 1 / (self.step + 1), self.alpha)
        for ema_param, param in zip(self.model.parameters(), model.parameters()):
            ema_param.data = decay * ema_param.data + (1 - decay) * param.data
        self.step += 1
def generate_class_mask(pseudo_labels, pseudo_labels2):
    labels = torch.unique(pseudo_labels)  # all unique labels
    labels_select = labels[torch.randperm(len(labels))][:len(labels) // 2]  # randomly select half of labels

    mask = (pseudo_labels.unsqueeze(-1) == labels_select).any(-1)
    return mask.float()


def generate_unsup_data(data, target, logits=None):
    batch_size, _, im_h, im_w = data.shape
    device = data.device

    new_data = []
    new_target = []
    new_logits = []
    for i in range(batch_size):
        mix_mask = generate_class_mask(target[i], target[(i + 1)% batch_size]).to(device)

        new_data.append((data[i] * mix_mask + data[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_target.append((target[i] * mix_mask + target[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))
        new_logits.append((logits[i] * mix_mask + logits[(i + 1) % batch_size] * (1 - mix_mask)).unsqueeze(0))

    new_data, new_target, new_logits = torch.cat(new_data), torch.cat(new_target), torch.cat(new_logits)
    return new_data, new_target.long() , new_logits

def compute_unsupervised_loss(predict, target, logits, strong_threshold):
    batch_size = predict.shape[0]
    valid_mask = (target >= 0).float()   # only count valid pixels

    weighting = logits.view(batch_size, -1).ge(strong_threshold).sum(-1) / valid_mask.view(batch_size, -1).sum(-1)
    loss = F.cross_entropy(predict, target, reduction='none', ignore_index=-1)
    weighted_loss = torch.mean(torch.masked_select(weighting[:, None, None] * loss, loss > 0))
    return weighted_loss
class Trainer(BaseTrainer):
    def __init__(self, model_stu, config, writer):
        self.model = model_stu
        self.model.train()
        self.ema = EMA(self.model,0.99)

        self.config = config
        self.writer = writer

    def entropy_loss(self, p):
        p = F.softmax(p, dim=1)
        log_p = F.log_softmax(p, dim=1)
        loss = -torch.sum(p * log_p, dim=1)
        return loss

    def dis_iter(self, batch):
        img_s, label_s, _, _, name = batch
        b, c, h, w = img_s.shape
        img_s = img_s.cuda()
        label_s = label_s.long().cuda()
        with torch.no_grad():
            pred_u, _ = self.ema.model(img_s)
            pseudo_logits, pseudo_labels = torch.max(torch.softmax(pred_u, dim=1), dim=1)
            pseudo_labels[label_s != 255] = label_s[label_s != 255]  # use selected class-balanced label
            pseudo_logits[label_s != 255] = 1.0
            train_u_aug_data, train_u_aug_label, train_u_aug_logits = \
                generate_unsup_data(img_s, pseudo_labels, pseudo_logits)
        pred_stu, feat_stu = self.model(train_u_aug_data)
        loss_s = compute_unsupervised_loss(pred_stu, train_u_aug_label, train_u_aug_logits, 0.97)
        loss = loss_s
        self.losses.loss_s = loss_s

        loss.backward()

    def iter(self, batch, r):
        img_s, label_s, _, _, name = batch
        b, c, h, w = img_s.shape
        pred_s = self.model.forward(img_s.cuda())[0]
        label_s = label_s.long().cuda()
        if self.config.method == 'simsfss':
            pred_s = pred_s.permute(0, 2, 3, 1).contiguous().view(-1, self.config.num_classes)
            pred_s_softmax = F.softmax(pred_s, -1)
            label_s = label_s.view(-1)
            loss_s = F.cross_entropy(pred_s, label_s, ignore_index=255)
            loss_e = self.entropy_loss(pred_s)
            loss_e = loss_e.mean()
            self.losses.loss_source = loss_s
            self.losses.loss_entropy = loss_e
            width = 3
            k = self.config.num_classes // 2 + random.randint(-width, width)
            _, labels_neg = torch.topk(pred_s_softmax, k, dim=1, sorted=True)
            s_neg = torch.log(torch.clamp(1. - pred_s_softmax, min=1e-5, max=1.))
            labels_neg = labels_neg[:, -1].squeeze().detach()
            loss_neg = criterion_nll(s_neg, labels_neg)
            self.losses.loss_neg = loss_neg
            loss = loss_s + 1 * loss_e + 1 * loss_neg
            loss.backward()

    def train(self):
        if self.config.neptune:
            neptune.init(project_qualified_name="solacex/segmentation-DA")
            neptune.create_experiment(params=self.config, name=self.config["note"])
        if self.config.resume:
            self.resume()
        else:
            self.round_start = 0

        for r in range(self.round_start, self.config.round):
            torch.manual_seed(1234)
            torch.cuda.manual_seed(1234)
            np.random.seed(1234)
            random.seed(1234)
            self.class_num = 19
            self.model = self.model.train()

            self.source_all = get_list(self.config.gta5.data_list)
            self.target_all = get_list(self.config.cityscapes.data_list)

            if self.config.method == 'ld' or self.config.method == "be":
                start = time.clock()
                print("cb_prop:{}".format(self.config.cb_prop))
                self.cb_thres = self.gene_thres(self.config.cb_prop)
                for i in self.cb_thres:
                    if self.cb_thres[i] > 0.999999999:
                        self.cb_thres[i] = 0.999999999
                print(self.cb_thres)
                self.save_pred(r)
                self.plabel_path = osp.join(self.config.plabel, self.config.note, str(r))
                # self.plabel_path = osp.join(self.config.plabel, self.config.note, str(r))
                end = time.clock()
                print('Running time: %s Seconds' % (end - start))
                self.config.cb_prop += 0.05
            else:
                self.plabel_path = None


            self.optim = torch.optim.SGD(
                self.model.optim_parameters(self.config.learning_rate),
                lr=self.config.learning_rate,
                momentum=self.config.momentum,
                weight_decay=self.config.weight_decay,
            )

            self.loader, _ = dataset.init_target_dataset(self.config, plabel_path=self.plabel_path,
                                                         target_selected=self.target_all)
            self.config.num_steps = 5000
            cu_iter = 0
            self.gamma = 1.0 * (r + 1)
            miou = self.validate()
            for epoch in range(self.config.epochs):
                for i_iter, batch in tqdm(enumerate(self.loader)):
                    # self.save_model('STNTHIA_source_only',0,0)
                    # print('done')
                    # self.model.module.disable_train()
                    # miou = self.validate()
                    cu_step = epoch * len(self.loader) + i_iter
                    self.model = self.model.train()
                    # self.model.module.enable_train()
                    self.losses = edict({})
                    self.optim.zero_grad()
                    adjust_learning_rate(self.optim, cu_step, self.config)
                    if self.config.method == 'ld':
                        self.iter(batch, r)
                    elif self.config.method == 'be':
                        self.dis_iter(batch)

                    self.optim.step()
                    self.ema.update(self.model)
                    if i_iter % self.config.print_freq == 0:
                        self.print_loss(i_iter)

                    if i_iter % self.config.val_freq == 0 and i_iter != 0:
                        # self.model.module.disable_train()
                        miou = self.validate()
                    if i_iter % self.config.save_freq == 0 and i_iter != 0:
                        self.save_model(self.config.source, cu_step, miou)
                miou = self.validate()
            self.config.learning_rate = self.config.learning_rate / (math.sqrt(2))
        if self.config.neptune:
            neptune.stop()

    def resume(self):
        iter_num = self.config.init_weight[-5]  # .split(".")[0].split("_")[1]
        iter_num = int(iter_num)
        self.round_start = int(math.ceil((iter_num + 1) / self.config.epochs))
        print("Resume from Round {}".format(self.round_start))
        if self.config.lr_decay == "sqrt":
            self.config.learning_rate = self.config.learning_rate / (
                    (math.sqrt(2)) ** self.round_start
            )

    def gene_thres(self, prop, num_cls=19):
        print('[Calculate Threshold using config.cb_prop]')  # r in section 3.3

        probs = {}
        freq = {}
        loader = dataset.init_test_dataset(self.config, self.config.target, set="train", selected=self.target_all,
                                           batchsize=1)
        for index, batch in tqdm(enumerate(loader)):
            img, label, _, _, _ = batch
            with torch.no_grad():
                pred = F.softmax(self.model.forward(img.cuda())[0], dim=1)
            pred_probs = pred.max(dim=1)[0]
            pred_probs = pred_probs.squeeze()
            pred_label = torch.argmax(pred, dim=1).squeeze()
            for i in range(num_cls):
                cls_mask = pred_label == i
                cnt = cls_mask.sum()
                if cnt == 0:
                    continue
                cls_probs = torch.masked_select(pred_probs, cls_mask)
                cls_probs = cls_probs.detach().cpu().numpy().tolist()
                cls_probs.sort()
                if i not in probs:
                    probs[i] = cls_probs[::5]  # reduce the consumption of memory
                else:
                    probs[i].extend(cls_probs[::5])

        growth = {}
        thres = {}
        for k in probs.keys():
            cls_prob = probs[k]
            cls_total = len(cls_prob)
            freq[k] = cls_total
            cls_prob = np.array(cls_prob)
            cls_prob = np.sort(cls_prob)
            index = int(cls_total * prop)
            cls_thres = cls_prob[-index]
            cls_thres2 = cls_prob[index]
            thres[k] = cls_thres
        print(thres)
        return thres

    def save_pred(self, round):
        # Using the threshold to generate pseudo labels and save  
        print("[Generate pseudo labels]")
        loader = dataset.init_test_dataset(self.config, self.config.target, set="train", selected=self.target_all)
        interp = nn.Upsample(size=(1024, 2048), mode="bilinear", align_corners=True)

        self.plabel_path = osp.join(self.config.plabel, self.config.note, str(round))

        mkdir(self.plabel_path)
        self.config.target_data_dir = self.plabel_path
        self.pool = Pool()  # save the probability of pseudo labels for the pixel-wise similarity matchinng, which is detailed around Eq. (9)
        accs = AverageMeter()  # Counter
        props = AverageMeter()  # Counter
        cls_acc = GroupAverageMeter()  # Class-wise Acc/Prop of Pseudo labels

        self.mean_memo = {i: [] for i in range(self.config.num_classes)}
        with torch.no_grad():
            for index, batch in tqdm(enumerate(loader)):
                image, label, _, _, name = batch
                label = label.cuda()
                img_name = name[0].split("/")[-1]
                dir_name = name[0].split("/")[0]
                img_name = img_name.replace("leftImg8bit", "gtFine_labelIds")
                temp_dir = osp.join(self.plabel_path, dir_name)
                if not os.path.exists(temp_dir):
                    os.mkdir(temp_dir)

                output = self.model.forward(image.cuda())[0]
                output = interp(output)
                # pseudo labels selected by glocal threshold
                mask, plabel = thres_cb_plabel(output, self.cb_thres, num_cls=self.config.num_classes)
                # pseudo labels selected by local threshold
                if round >= 0:
                    local_prop = self.config.cb_prop
                    mask2, plabel2 = gene_plabel_prop(output, local_prop)
                    mask, plabel = mask_fusion(output, mask, mask2)
                self.pool.update_pool(output, mask=mask.float())
                acc, prop, cls_dict = Acc(plabel, label, num_cls=self.config.num_classes)
                cnt = (plabel != 255).sum().item()
                accs.update(acc, cnt)
                props.update(prop, 1)
                cls_acc.update(cls_dict)
                plabel = plabel.view(1024, 2048)
                plabel = plabel.cpu().numpy()

                plabel = np.asarray(plabel, dtype=np.uint8)
                plabelz = Image.fromarray(plabel)
                plabelz.save("%s/%s.png" % (temp_dir, img_name.split(".")[0]))
        print('The Accuracy :{:.2%} and proportion :{:.2%} of Pseudo Labels'.format(accs.avg.item(), props.avg.item()))
        if self.config.neptune:
            neptune.send_metric("Acc", accs.avg)
            neptune.send_metric("Prop", props.avg)



    def save_model(self, source, iter, miou):
        name = str(iter) + "_miou" + str(miou)
        tmp_name = "_EntMin_".join((source, str(name))) + ".pth"
        torch.save(self.model.state_dict(), osp.join(self.config["snapshot"], tmp_name))
