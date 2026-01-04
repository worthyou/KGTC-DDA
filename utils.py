import csv
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

class MetricLogger(object):
    def __init__(self, attr_names, parse_formats, save_path):
        self._attr_format_dict = OrderedDict(zip(attr_names, parse_formats))
        self._file = open(save_path, 'w')
        self._csv = csv.writer(self._file)
        self._csv.writerow(attr_names)
        self._file.flush()

    def log(self, **kwargs):
        self._csv.writerow([parse_format % kwargs[attr_name]
                            for attr_name, parse_format in self._attr_format_dict.items()])
        self._file.flush()

    def close(self):
        self._file.close()

# 自定义 Focal Loss 类
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # 正类权重
        self.gamma = gamma  # 聚焦参数
        self.reduction = reduction  # 'mean', 'sum', 或 'none'

    def forward(self, inputs, targets):
        # inputs 是 logits，targets 是 0/1 标签
        p = th.sigmoid(inputs)  # 将 logits 转换为概率
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')  # 计算基础交叉熵损失
        p_t = p * targets + (1 - p) * (1 - targets)  # p_t 是真实类别的概率
        loss = ce_loss * ((1 - p_t) ** self.gamma)  # 添加调制因子 (1 - p_t)^γ

        # 应用 alpha 权重
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        loss = alpha_t * loss

        # 根据 reduction 参数聚合损失
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss