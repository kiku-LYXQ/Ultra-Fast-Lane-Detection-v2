from utils.loss import SoftmaxFocalLoss, ParsingRelationLoss, ParsingRelationDis, MeanLoss, TokenSegLoss, VarLoss, EMDLoss, RegLoss
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, Mae
from utils.dist_utils import DistSummaryWriter

import torch


def get_optimizer(net,cfg):
    training_params = filter(lambda p: p.requires_grad, net.parameters())
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(training_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(training_params, lr=cfg.learning_rate, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def get_scheduler(optimizer, cfg, iters_per_epoch):
    if cfg.scheduler == 'multi':
        scheduler = MultiStepLR(optimizer, cfg.steps, cfg.gamma, iters_per_epoch, cfg.warmup, iters_per_epoch if cfg.warmup_iters is None else cfg.warmup_iters)
    elif cfg.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, cfg.epoch * iters_per_epoch, eta_min = 0, warmup = cfg.warmup, warmup_iters = cfg.warmup_iters)
    else:
        raise NotImplementedError
    return scheduler

def get_loss_dict(cfg):
    if cfg.dataset == 'CurveLanes':
        loss_dict = {
            'name': ['cls_loss', 'relation_loss', 'relation_dis','cls_loss_col','cls_ext','cls_ext_col', 'mean_loss_row', 'mean_loss_col','var_loss_row', 'var_loss_col', 'lane_token_seg_loss_row', 'lane_token_seg_loss_col'],
            'op': [SoftmaxFocalLoss(2, ignore_lb=-1), ParsingRelationLoss(), ParsingRelationDis(), SoftmaxFocalLoss(2, ignore_lb=-1), torch.nn.CrossEntropyLoss(),  torch.nn.CrossEntropyLoss(), MeanLoss(), MeanLoss(), VarLoss(cfg.var_loss_power), VarLoss(cfg.var_loss_power), TokenSegLoss(), TokenSegLoss()],
            'weight': [1.0, cfg.sim_loss_w, cfg.shp_loss_w, 1.0, 1.0, 1.0, cfg.mean_loss_w, cfg.mean_loss_w, 0.01, 0.01, 1.0, 1.0],
            'data_src': [('cls_out', 'cls_label'), ('cls_out',), ('cls_out',), ('cls_out_col', 'cls_label_col'), 
            ('cls_out_ext', 'cls_out_ext_label'), ('cls_out_col_ext', 'cls_out_col_ext_label') , ('cls_out', 'cls_label'),('cls_out_col', 'cls_label_col'),('cls_out', 'cls_label'),('cls_out_col', 'cls_label_col'), ('seg_out_row', 'seg_label'), ('seg_out_col', 'seg_label')
            ],
        }
    elif cfg.dataset in ['Tusimple', 'CULane']:
        loss_dict = {
            'name': ['cls_loss', 'relation_loss', 'relation_dis','cls_loss_col','cls_ext','cls_ext_col', 'mean_loss_row', 'mean_loss_col'],
            'op': [SoftmaxFocalLoss(2, ignore_lb=-1), ParsingRelationLoss(), ParsingRelationDis(), SoftmaxFocalLoss(2, ignore_lb=-1), torch.nn.CrossEntropyLoss(),  torch.nn.CrossEntropyLoss(), MeanLoss(), MeanLoss(),],
            'weight': [1.0, cfg.sim_loss_w, cfg.shp_loss_w, 1.0, 1.0, 1.0, cfg.mean_loss_w, cfg.mean_loss_w,],
            'data_src': [('cls_out', 'cls_label'), ('cls_out',), ('cls_out',), ('cls_out_col', 'cls_label_col'), 
            ('cls_out_ext', 'cls_out_ext_label'), ('cls_out_col_ext', 'cls_out_col_ext_label') , ('cls_out', 'cls_label'),('cls_out_col', 'cls_label_col'),
            ],
        }
    else:
        raise NotImplementedError

    
    if cfg.use_aux:
        loss_dict['name'].append('seg_loss')
        loss_dict['op'].append(torch.nn.CrossEntropyLoss(weight = torch.tensor([0.6, 1., 1., 1., 1.])).cuda())
        loss_dict['weight'].append(1.0)
        loss_dict['data_src'].append(('seg_out', 'seg_label'))

    assert len(loss_dict['name']) == len(loss_dict['op']) == len(loss_dict['data_src']) == len(loss_dict['weight'])
    return loss_dict

def get_metric_dict(cfg):
    """创建模型评估指标字典
    核心设计原则：
    1. 多任务联合评估：同时评估行/列的位置预测和存在性判断
    2. 动态扩展机制：根据配置灵活添加辅助任务指标
    3. 数据流映射：明确每个指标对应的模型输出和标签数据
    """
    # 初始化基础指标（行方向相关指标）
    metric_dict = {
        'name': ['top1', 'top2', 'top3', 'ext_row', 'ext_col'],
        'op': [
            AccTopk(-1, 1),  # 行位置预测Top1准确率（dim=-1表示自动选择分类维度）
            AccTopk(-1, 2),  # 行位置预测Top2准确率
            AccTopk(-1, 3),  # 行位置预测Top3准确率
            MultiLabelAcc(), # 行存在性多标签准确率（每个位置二分类）
            MultiLabelAcc()  # 列存在性多标签准确率
        ],
        'data_src': [
            # 数据源格式：(模型输出字段名, 标签字段名)
            ('cls_out', 'cls_label'),       # 行位置预测结果
            ('cls_out', 'cls_label'),       # 同一数据源复用
            ('cls_out', 'cls_label'),
            ('cls_out_ext', 'cls_out_ext_label'),    # 行存在性预测
            ('cls_out_col_ext','cls_out_col_ext_label') # 列存在性预测
        ]
    }

    # 添加列方向评估指标 ------------------------------------------------------
    # 扩展指标名称（与行指标对称设计）
    metric_dict['name'].extend(['col_top1', 'col_top2', 'col_top3'])
    # 使用相同的评估类但不同数据源
    metric_dict['op'].extend([
        AccTopk(-1, 1),  # 列Top1准确率（dim参数自动适应输出维度）
        AccTopk(-1, 2),  # 列Top2准确率
        AccTopk(-1, 3)   # 列Top3准确率
    ])
    # 指定列方向预测的数据源
    metric_dict['data_src'].extend([
        ('cls_out_col', 'cls_label_col'),  # 列位置预测结果
        ('cls_out_col', 'cls_label_col'),
        ('cls_out_col', 'cls_label_col')
    ])

    # 添加辅助任务指标（当启用分割头时）-----------------------------------------
    if cfg.use_aux:
        metric_dict['name'].append('iou')  # 分割任务评估指标
        metric_dict['op'].append(
            Metric_mIoU(5)  # 基于5个IoU计算周期的移动平均
        )
        metric_dict['data_src'].append(
            ('seg_out', 'seg_label')  # 分割输出与分割标签
        )

    # 完整性校验（确保三个列表长度一致）
    assert len(metric_dict['name']) == len(metric_dict['op']) == len(metric_dict['data_src'])
    return metric_dict



class MultiStepLR:
    def __init__(self, optimizer, steps, gamma = 0.1, iters_per_epoch = None, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.steps = steps
        self.steps.sort()
        self.gamma = gamma
        self.iters_per_epoch = iters_per_epoch
        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return
        
        # multi policy
        if self.iters % self.iters_per_epoch == 0:
            epoch = int(self.iters / self.iters_per_epoch)
            power = -1
            for i, st in enumerate(self.steps):
                if epoch < st:
                    power = i
                    break
            if power == -1:
                power = len(self.steps)
            # print(self.iters, self.iters_per_epoch, self.steps, power)
            
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * (self.gamma ** power)
import math
class CosineAnnealingLR:
    def __init__(self, optimizer, T_max , eta_min = 0, warmup = None, warmup_iters = None):
        self.warmup = warmup
        self.warmup_iters = warmup_iters
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min

        self.iters = 0
        self.base_lr = [group['lr'] for group in optimizer.param_groups]

    def step(self, external_iter = None):
        self.iters += 1
        if external_iter is not None:
            self.iters = external_iter
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate
            return
        
        # cos policy

        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.iters / self.T_max)) / 2

        