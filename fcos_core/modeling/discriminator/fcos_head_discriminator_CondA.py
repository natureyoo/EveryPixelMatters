import torch
import torch.nn.functional as F
from torch import nn

from .layer import GradientReversal


# class FCOSDiscriminator_CondA(nn.Module):
#     def __init__(self, num_convs=2, in_channels=256, grad_reverse_lambda=-1.0, center_aware_weight=0.0, center_aware_type='ca_loss', grl_applied_domain='both'):
#         """
#         Arguments:
#             in_channels (int): number of channels of the input feature
#         """
#         super(FCOSDiscriminator_CondA, self).__init__()
#
#         dis_tower = []
#         for i in range(num_convs):
#             if i == 0:
#                 dis_tower.append(
#                     nn.Conv2d(
#                         # in_channels + 8 + 4,
#                         in_channels * 8,
#                         in_channels,
#                         kernel_size=3,
#                         stride=1,
#                         padding=1
#                     )
#                 )
#             else:
#                 dis_tower.append(
#                     nn.Conv2d(
#                         in_channels,
#                         in_channels,
#                         kernel_size=3,
#                         stride=1,
#                         padding=1
#                     )
#                 )
#             dis_tower.append(nn.GroupNorm(32, in_channels))
#             dis_tower.append(nn.ReLU())
#
#         self.add_module('dis_tower', nn.Sequential(*dis_tower))
#
#         self.cls_logits = nn.Conv2d(
#             in_channels, 1, kernel_size=3, stride=1,
#             padding=1
#         )
#         # self.BN_cls = nn.BatchNorm2d(8)
#         # self.BN_reg = nn.BatchNorm2d(4)
#
#         # initialization
#         for modules in [self.dis_tower, self.cls_logits]:
#             for l in modules.modules():
#                 if isinstance(l, nn.Conv2d):
#                     torch.nn.init.normal_(l.weight, std=0.01)
#                     torch.nn.init.constant_(l.bias, 0)
#
#         self.grad_reverse = GradientReversal(grad_reverse_lambda)
#         self.loss_fn = nn.BCEWithLogitsLoss()
#         self.loss_fn_no_reduce = nn.BCEWithLogitsLoss(reduction='none')
#
#         # hyperparameters
#         assert center_aware_type == 'ca_loss' or center_aware_type == 'ca_feature'
#         self.center_aware_weight = center_aware_weight
#         self.center_aware_type = center_aware_type
#
#         assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
#         self.grl_applied_domain = grl_applied_domain
#
#     def forward(self, feature, target, score_map=None, domain='source'):
#         assert target == 0 or target == 1 or target == 0.1 or target == 0.9
#         assert domain == 'source' or domain == 'target'
#
#         # Generate cneter-aware map
#         # box_cls_map = score_map["box_cls"]
#         centerness_map = score_map["centerness"].clone().sigmoid()
#         box_regression_map = score_map["box_regression"].detach()
#
#         ############ feature concat ############
#         # box_cls_map = self.BN_cls(box_cls_map)
#         # box_regression_map = self.BN_reg(box_regression_map)
#         # feature = torch.cat((feature, box_cls_map, box_regression_map), 1) * centerness_map
#
#         ############ feature * class outer product ############
#         # box_cls_pred = nn.Softmax(dim=1)(box_cls_map).detach()
#         # entropy = -torch.sum(box_cls_pred * torch.log(box_cls_pred), dim=1)
#         # entropy = 1 + torch.exp(-entropy)
#         # sh = feature.shape
#         # feature = feature.permute(0,2,3,1).reshape(-1,256)
#         # box_cls_pred = box_cls_pred.permute(0,2,3,1).reshape(-1, 8)
#         # feature = torch.bmm(feature.unsqueeze(2), box_cls_pred.unsqueeze(1))
#         # feature = feature.reshape(feature.shape[0], -1).reshape(sh[0], sh[2], sh[3], -1).permute(0,3,1,2)
#         # feature = feature * centerness_map * entropy.unsqueeze(1)
#
#         ############ feature * regression binning outer product ############
#         # import pdb; pdb.set_trace()
#         # [0 ** 2, 32 ** 2],  # small
#         # [32 ** 2, 96 ** 2],  # medium
#         # [96 ** 2, 1e5 ** 2],  # large
#         # [96 ** 2, 128 ** 2],  # 96-128
#         # [128 ** 2, 256 ** 2],  # 128-256
#         # [256 ** 2, 512 ** 2],  # 256-512
#         # [512 ** 2, 1e5 ** 2],
#         bin_mean = torch.tensor([32, 96, 128, 256, 512]).cuda()
#         bin_std = torch.tensor([16, 24, 32, 48, 64]).cuda()
#         # bbox_sh = box_regression_map.shape
#
#         box_regression_map = box_regression_map.permute(0,2,3,1).reshape(-1, 4)
#         box_prob = torch.exp(-(box_regression_map.unsqueeze(-1) - bin_mean.unsqueeze(0,1)) ** 2/2**bin_std.unsqueeze(0,1)**2)
#         box_prob = box_prob / torch.sum(box_prob, dim=-1)
#
#         feature = feature.permute(0,2,3,1).reshape(-1,256)
#         feature = torch.bmm(feature.unsqueeze(2), box_prob[:,0,:].unsqueeze(1))
#         feature = feature.reshape(feature.shape[0], -1).reshape(sh[0], sh[2], sh[3], -1).permute(0,3,1,2)
#         feature = feature * centerness_map
#
#         if self.grl_applied_domain == 'both':
#             feature = self.grad_reverse(feature)
#         elif self.grl_applied_domain == 'target':
#             if domain == 'target':
#                 feature = self.grad_reverse(feature)
#
#         # Forward
#         x = self.dis_tower(feature)
#         x = self.cls_logits(x)
#
#         target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
#         loss = self.loss_fn(x, target)
#
#         return loss


class FCOSDiscriminator_CondA(nn.Module):
    def __init__(self, num_convs=2, in_channels=256, grad_reverse_lambda=-1.0, center_aware_weight=0.0, center_aware_type='ca_loss', grl_applied_domain='both'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator_CondA, self).__init__()

        self.embed_dim = in_channels
        # self.random_embed = torch.randn(in_channels, self.embed_dim).cuda()
        self.random_embed = None
        dis_tower = {}
        self.cls_logits = {}
        self.num = {'class':8, 'reg_l':3, 'reg_t':3, 'reg_b':3, 'reg_r':3}
        # for key in ['class', 'reg_l', 'reg_t', 'reg_b', 'reg_r']:
        for key in ['reg_l']:
            dis_tower[key] = []
            for i in range(num_convs):
                if i == 0:
                    dis_tower[key].append(
                        nn.Conv2d(
                            self.embed_dim * self.num[key],
                            self.embed_dim,
                            kernel_size=3,
                            stride=1,
                            padding=1
                        )
                    )
                else:
                    dis_tower[key].append(
                        nn.Conv2d(
                            self.embed_dim,
                            self.embed_dim,
                            kernel_size=3,
                            stride=1,
                            padding=1
                        )
                    )
                dis_tower[key].append(nn.GroupNorm(self.embed_dim, self.embed_dim))
                dis_tower[key].append(nn.ReLU())

            self.add_module('dis_tower_{}'.format(key), nn.Sequential(*dis_tower[key]))
            self.add_module('cls_logits_{}'.format(key), nn.Conv2d(
                self.embed_dim, 1, kernel_size=3, stride=1,
                padding=1
            ))

            # self.cls_logits[key] = nn.Conv2d(
            #     self.embed_dim, 1, kernel_size=3, stride=1,
            #     padding=1
            # )
        # initialization
        # for modules in [self.dis_tower_class, self.dis_tower_reg_l, self.dis_tower_reg_t, self.dis_tower_reg_b, self.dis_tower_reg_r]:
        for modules in [self.dis_tower_reg_l]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        # for modules in [self.cls_logits[key] for key in ['class', 'reg_l', 'reg_t', 'reg_b', 'reg_r']]:
        # for modules in [self.cls_logits_class, self.cls_logits_reg_l, self.cls_logits_reg_t, self.cls_logits_reg_b, self.cls_logits_reg_r]:
        for modules in [self.cls_logits_reg_l]:
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

    def forward(self, feature, target, score_map=None, domain='source'):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'

        # Generate cneter-aware map
        # box_cls_map = score_map["box_cls"]
        box_cls_map = score_map["box_cls"].clone().sigmoid()
        centerness_map = score_map["centerness"].clone().sigmoid()
        box_cls_pred = box_cls_map.detach()
        box_regression_map = score_map["box_regression"]
        # print(feature)
        # print(feature.min(), feature.max(), feature.mean())
        sh = feature.shape
        feature = feature.permute(0,2,3,1).reshape(-1,256)
        if self.random_embed is not None:
            feature = torch.mm(feature, self.random_embed)
            feature = feature / self.embed_dim ** 0.5
            # print(self.random_embed)
            # print(feature.min(), feature.max(), feature.mean())
        feature_ = {}
        loss = 0

        n, c, h, w = box_cls_map.shape
        maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
        box_cls_map = maxpooling(box_cls_map)
        # Normalize the center-aware map
        atten_map = self.center_aware_weight * box_cls_map * centerness_map

        ############ feature * class outer product ############
        # entropy = - box_cls_pred * torch.log(box_cls_pred + 1e-7)
        # entropy = 1 + torch.exp(-entropy)

        # box_cls_pred = box_cls_pred.permute(0,2,3,1).reshape(-1, 8)
        # feature_['class'] = torch.bmm(feature.unsqueeze(2), box_cls_pred.unsqueeze(1))
        # feature_['class'] = feature_['class'].reshape(feature_['class'].shape[0], -1).reshape(sh[0], sh[2], sh[3], -1).permute(0,3,1,2)
        # # feature_['class'] = feature_['class'] * centerness_map * entropy.unsqueeze(1)
        # feature_['class'] = feature_['class'] * atten_map

        ############ feature * regression binning outer product ############
        # import pdb; pdb.set_trace()
        # [0 ** 2, 32 ** 2],  # small
        # [32 ** 2, 96 ** 2],  # medium
        # [96 ** 2, 1e5 ** 2],  # large
        # [96 ** 2, 128 ** 2],  # 96-128
        # [128 ** 2, 256 ** 2],  # 128-256
        # [256 ** 2, 512 ** 2],  # 256-512
        # [512 ** 2, 1e5 ** 2],
        # bin_mean = torch.tensor([32, 96, 128, 256, 512]).cuda()
        # bin_std = torch.tensor([16, 24, 32, 48, 64]).cuda()
        bin_mean = torch.tensor([32, 64, 128]).cuda()
        bin_std = torch.tensor([16, 16, 16]).cuda()
        # bbox_sh = box_regression_map.shape
        # import pdb
        box_regression_map = box_regression_map.permute(0,2,3,1).reshape(-1, 4)     # (528,4)
        # print(box_regression_map[:,0])
        # print(box_regression_map.unsqueeze(-1).shape)
        # print(bin_mean.reshape(1,1,-1).shape)
        # box_prob = torch.exp(-(box_regression_map.unsqueeze(-1) - bin_mean.reshape(1,1,-1)) ** 2/(2*bin_std.reshape(1,1,-1)**2))
        # box_prob = box_prob / torch.sum(box_prob, dim=-1).unsqueeze(-1)
        box_cls_gt = (box_regression_map.unsqueeze(-1) - bin_mean.reshape(1,1,-1)) ** 2/(2*bin_std.reshape(1,1,-1)**2)
        box_cls_gt = torch.argmin(box_cls_gt, dim=-1)
        box_cls_onehot = torch.FloatTensor(box_cls_gt.shape[0], 3).cuda()
        box_cls_onehot.zero_()
        box_cls_onehot.scatter_(1, box_cls_gt[:, 0].reshape(-1, 1), 1)
        # print(box_prob.shape)
        # print(torch.sum(box_prob, dim=-1))
        # reg_cls = torch.argmax(box_prob, dim=-1)
        # print(torch.sum(reg_cls == 0))
        # print(torch.sum(reg_cls == 1))
        # print(torch.sum(reg_cls == 2))
        # print(torch.sum(reg_cls == 3))
        # print(torch.sum(reg_cls == 4))
        # print(torch.sum(box_prob, dim=-1))
        # for idx, key in enumerate(['reg_l', 'reg_t', 'reg_b', 'reg_r']):
        for idx, key in enumerate(['reg_l']):
            feature_[key] = torch.bmm(feature.unsqueeze(2), box_cls_onehot.unsqueeze(1))
            feature_[key] = feature_[key].reshape(feature_[key].shape[0], -1).reshape(sh[0], sh[2], sh[3], -1).permute(0,3,1,2)
            # feature_[key] = feature_[key] * centerness_map * entropy.unsqueeze(1)
            feature_[key] = feature_[key] * atten_map

        # for key in ['class', 'reg_l', 'reg_t', 'reg_b', 'reg_r']:
        for key in ['reg_l']:
            if self.grl_applied_domain == 'both':
                feature_[key] = self.grad_reverse(feature_[key])
            elif self.grl_applied_domain == 'target':
                if domain == 'target':
                    feature_[key] = self.grad_reverse(feature_[key])

            # Forward
            if key == 'class':
                x = self.dis_tower_class(feature_[key])
                x = self.cls_logits_class(x)
            elif key == 'reg_l':
                x = self.dis_tower_reg_l(feature_[key])
                x = self.cls_logits_reg_l(x)
            elif key == 'reg_t':
                x = self.dis_tower_reg_t(feature_[key])
                x = self.cls_logits_reg_t(x)
            elif key == 'reg_b':
                x = self.dis_tower_reg_b(feature_[key])
                x = self.cls_logits_reg_b(x)
            else:
                x = self.dis_tower_reg_r(feature_[key])
                x = self.cls_logits_reg_r(x)
            # x = self.cls_logits[key](x)
            if torch.abs(torch.mean(x)) > 10*10:
                print(feature)
                print(key)
                print(feature_[key])
                print(x)
                import pdb; pdb.set_trace()
            # print(x.shape)
            target_ = torch.full(x.shape, target, dtype=torch.float, device=x.device)
            if key == 'class':
                # loss += self.loss_fn(x, target_)
                loss += torch.mean((x - target_) ** 2)
            else:
                # loss += self.loss_fn(x, target_) * 0.1
                loss += torch.mean((x - target_) ** 2)
        return loss
