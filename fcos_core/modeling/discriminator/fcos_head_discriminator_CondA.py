import torch
import torch.nn.functional as F
from torch import nn

from .layer import GradientReversal


class FCOSDiscriminator_CondA(nn.Module):
    def __init__(self, num_convs=2, in_channels=256, grad_reverse_lambda=-1.0, center_aware_weight=0.0, \
                 center_aware_type='ca_loss', grl_applied_domain='both', class_align=False, reg_left_align=False, \
                 reg_top_align=False, expand_dim=0, level='P3'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator_CondA, self).__init__()

        self.embed_dim = in_channels
        # self.random_embed = torch.randn(in_channels, self.embed_dim).cuda()
        self.random_embed = None
        dis_tower = []
        self.num = {'class':8, 'reg_l':3, 'reg_t':3, 'reg_b':3, 'reg_r':3}
        self.class_align = class_align
        self.reg_left_align = reg_left_align
        self.reg_top_align = reg_top_align
        # self.bin_mean = torch.tensor([32, 64, 128]).cuda()
        if level == 'P3':
            # self.bin_mean = torch.tensor([4,12,20]).cuda()
            self.bin_mean = torch.tensor([12,20,44]).cuda()
        elif level == 'P4':
            self.bin_mean = torch.tensor([12,20,44]).cuda()
            # self.bin_mean = torch.tensor([20,44,84]).cuda()
        elif level == 'P5':
            self.bin_mean = torch.tensor([20,44,84]).cuda()
            # self.bin_mean = torch.tensor([44,84,172]).cuda()
        elif level == 'P6':
            self.bin_mean = torch.tensor([44,84,172]).cuda()
            # self.bin_mean = torch.tensor([84,172,340]).cuda()
        else:
            self.bin_mean = torch.tensor([44,84,172]).cuda()
            # self.bin_mean = torch.tensor([84,172,340]).cuda()
        self.bin_std = self.bin_mean / 4
        self.stat = {'source': [1/3, 1/3, 1/3], 'target': [1/3, 1/3, 1/3]}
        if expand_dim > 0:
            self.expand = True
            self.outer_num = expand_dim
        else:
            self.expand = False
            self.outer_num = 0
            if self.class_align:
                self.outer_num += 8
            if self.reg_left_align:
                self.outer_num += 3
            if self.reg_top_align:
                self.outer_num += 3

        assert self.outer_num > 0

        for i in range(num_convs):
            if i == 0:
                dis_tower.append(
                    nn.Conv2d(
                        self.embed_dim * self.outer_num,
                        self.embed_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
            else:
                dis_tower.append(
                    nn.Conv2d(
                        self.embed_dim,
                        self.embed_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
            dis_tower.append(nn.GroupNorm(self.embed_dim, self.embed_dim))
            dis_tower.append(nn.ReLU())

        self.add_module('dis_tower', nn.Sequential(*dis_tower))
        self.add_module('cls_logits', nn.Conv2d(
            self.embed_dim, 1, kernel_size=3, stride=1,
            padding=1
        ))

        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn_no_reduce = nn.BCEWithLogitsLoss(reduction='none')

        # hyperparameters
        assert center_aware_type == 'ca_loss' or center_aware_type == 'ca_feature'
        self.center_aware_weight = center_aware_weight
        self.center_aware_type = center_aware_type

        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.grl_applied_domain = grl_applied_domain

    def forward(self, feature, target, score_map=None, domain='source', alpha=0.0):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        # Generate cneter-aware map
        box_cls_map = score_map["box_cls"].clone().sigmoid()
        centerness_map = score_map["centerness"].clone().sigmoid()
        box_cls_pred = box_cls_map.detach()
        box_regression_map = score_map["box_regression"]
        sh = feature.shape
        feature = feature.permute(0,2,3,1).reshape(-1, self.embed_dim)

        if self.random_embed is not None:
            feature = torch.mm(feature, self.random_embed)
            feature = feature / self.embed_dim ** 0.5

        feature_ = {}
        n, c, h, w = box_cls_map.shape
        maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
        box_cls_map = maxpooling(box_cls_map)
        # Normalize the center-aware map
        # atten_map = self.center_aware_weight * box_cls_map * centerness_map
        atten_map = self.center_aware_weight * box_cls_map
        ############ feature dimension expand ############
        if self.expand:
            feature = torch.bmm(feature.unsqueeze(2), torch.ones(n*h*w, 1, self.outer_num).cuda() / self.outer_num)
            feature = feature.reshape(n, h, w, -1).permute(0, 3, 1, 2) * atten_map
        else:
            ############ feature * class outer product ############
            if self.class_align:
                # entropy = - box_cls_pred * torch.log(box_cls_pred + 1e-7)
                # entropy = 1 + torch.exp(-entropy)

                box_cls_pred = box_cls_pred.permute(0,2,3,1).reshape(-1, 8)
                box_cls_pred = alpha * torch.ones(box_cls_pred.shape).cuda() / 8 + (1 - alpha) * box_cls_pred
                feature_['class'] = torch.bmm(feature.unsqueeze(2), box_cls_pred.unsqueeze(1))
                feature_['class'] = feature_['class'].reshape(feature_['class'].shape[0], -1).reshape(sh[0], sh[2], sh[3], -1).permute(0,3,1,2)
                # feature_['class'] = feature_['class'] * centerness_map * entropy.unsqueeze(1)
                feature_['class'] = feature_['class'] * atten_map

            ############ feature * regression binning outer product ############
            if self.reg_left_align or self.reg_top_align:
                box_regression_map = box_regression_map.permute(0,2,3,1).reshape(-1, 4)     # (528,4)
                # box_cls_gt = (box_regression_map.unsqueeze(-1) - self.bin_mean.reshape(1,1,-1)) ** 2/(2*self.bin_std.reshape(1,1,-1)**2)
                box_cls_gt = (box_regression_map.unsqueeze(-1) - self.bin_mean.reshape(1,1,-1)) ** 2
                # box_cls_gt = torch.argmin(box_cls_gt, dim=-1)
                if self.reg_left_align:
                    # box_cls_onehot = torch.FloatTensor(box_cls_gt.shape[0], 3).cuda()
                    # box_cls_onehot.zero_()
                    # box_cls_onehot.scatter_(1, box_cls_gt[:, 0].reshape(-1, 1), 1)
                    # box_cls_onehot = alpha * torch.ones(box_cls_onehot.shape).cuda() / 3 + (1 - alpha) * box_cls_onehot
                    box_cls_onehot = - box_cls_gt[:, 0] / self.bin_std.reshape(1, -1) ** 2
                    box_cls_onehot = torch.nn.Softmax(dim=1)(box_cls_onehot)
                    stat = (atten_map.permute(0,2,3,1).reshape(-1,1) * box_cls_onehot).sum(dim=0)
                    stat /= stat.sum()
                    for bin_idx in range(3):
                        self.stat[domain][bin_idx] = 0.99 * self.stat[domain][bin_idx] + 0.01 * stat[bin_idx].item()
                    feature_['reg_l'] = torch.bmm(feature.unsqueeze(2), box_cls_onehot.unsqueeze(1))
                    feature_['reg_l'] = feature_['reg_l'].reshape(feature_['reg_l'].shape[0], -1).reshape(sh[0], sh[2], sh[3], -1).permute(0,3,1,2)
                    # feature_[key] = feature_[key] * centerness_map * entropy.unsqueeze(1)
                    feature_['reg_l'] = feature_['reg_l'] * atten_map
                if self.reg_top_align:
                    # box_cls_onehot = torch.FloatTensor(box_cls_gt.shape[0], 3).cuda()
                    # box_cls_onehot.zero_()
                    # box_cls_onehot.scatter_(1, box_cls_gt[:, 1].reshape(-1, 1), 1)
                    # box_cls_onehot = alpha * torch.ones(box_cls_onehot.shape).cuda() / 3 + (1 - alpha) * box_cls_onehot
                    box_cls_onehot = - box_cls_gt[:, 1] / self.bin_std.reshape(1,-1) ** 2
                    box_cls_onehot = torch.nn.Softmax(dim=1)(box_cls_onehot)
                    stat = (atten_map.permute(0,2,3,1).reshape(-1,1) * box_cls_onehot).sum(dim=0)
                    stat /= stat.sum()
                    for bin_idx in range(3):
                        self.stat[domain][bin_idx] = 0.99 * self.stat[domain][bin_idx] + 0.01 * stat[bin_idx].item()
                    if domain == 'source':
                        box_cls_onehot = box_cls_onehot * (torch.tensor(self.stat['target']) / torch.tensor(self.stat['source'])).unsqueeze(0).cuda()
                    feature_['reg_t'] = torch.bmm(feature.unsqueeze(2), box_cls_onehot.unsqueeze(1))
                    feature_['reg_t'] = feature_['reg_t'].reshape(feature_['reg_t'].shape[0], -1).reshape(sh[0], sh[2], sh[3], -1).permute(0,3,1,2)
                    # feature_[key] = feature_[key] * centerness_map * entropy.unsqueeze(1)
                    feature_['reg_t'] = feature_['reg_t'] * atten_map

            feature = torch.cat([feature_[k] for k in feature_.keys()], dim=1)

        if self.grl_applied_domain == 'both':
            feature = self.grad_reverse(feature)
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                feature = self.grad_reverse(feature)

        x = self.dis_tower(feature)
        x = self.cls_logits(x)

        target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
        # loss = self.loss_fn(x, target)
        loss = torch.mean((x - target) ** 2)

        return loss, self.stat[domain]
