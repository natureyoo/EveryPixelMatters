import torch
import torch.nn.functional as F
from torch import nn

from .layer import GradientReversal


class FCOSDiscriminator_HA(nn.Module):
    def __init__(self, num_convs=2, in_channels=12, grad_reverse_lambda=-1.0, center_aware_weight=0.0, grl_applied_domain='both'):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(FCOSDiscriminator_HA, self).__init__()
        dis_tower = []
        for i in range(num_convs):
            dis_tower.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1))
            dis_tower.append(nn.GroupNorm(in_channels, in_channels))
            dis_tower.append(nn.ReLU())
        self.add_module('dis_tower', nn.Sequential(*dis_tower))
        self.cls_logits = nn.Conv2d(in_channels, 1, kernel_size=3, stride=1, padding=1)
        # initialization
        for modules in [self.dis_tower, self.cls_logits]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)
        self.grad_reverse = GradientReversal(grad_reverse_lambda)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn_no_reduce = nn.BCEWithLogitsLoss(reduction='none')
        assert grl_applied_domain == 'both' or grl_applied_domain == 'target'
        self.center_aware_weight = center_aware_weight
        self.grl_applied_domain = grl_applied_domain

    def forward(self, target, score_map=None, domain='source'):
        assert target == 0 or target == 1 or target == 0.1 or target == 0.9
        assert domain == 'source' or domain == 'target'
        # Generate cneter-aware map
        box_cls_map = score_map["box_cls"]
        centerness_map = score_map["centerness"].sigmoid()
        box_regression_map = score_map["box_regression"]
        box_cls_regression_map = torch.cat((box_cls_map, box_regression_map), 1) * centerness_map
        # n, c, h, w = box_cls_map.shape
        # maxpooling = nn.AdaptiveMaxPool3d((1, h, w))
        # box_cls_map_pool = maxpooling(box_cls_map.sigmoid())
        # atten_map = (self.center_aware_weight * box_cls_map_pool * centerness_map).sigmoid()
        # box_cls_regression_map = torch.cat((box_cls_map, box_regression_map), 1) * atten_map
        if self.grl_applied_domain == 'both':
            output = self.grad_reverse(box_cls_regression_map)
        elif self.grl_applied_domain == 'target':
            if domain == 'target':
                output = self.grad_reverse(box_cls_regression_map)
        # Forward
        x = self.dis_tower(output)
        x = self.cls_logits(x)
        target = torch.full(x.shape, target, dtype=torch.float, device=x.device)
        loss = self.loss_fn(x, target)
        return loss
