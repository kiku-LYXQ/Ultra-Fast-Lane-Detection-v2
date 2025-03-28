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
    """
    构建多任务损失函数字典，支持不同数据集和辅助任务配置
    参数:
        cfg (Config): 包含以下属性的配置对象:
            - dataset: 数据集名称 ['CurveLanes', 'Tusimple', 'CULane']
            - sim_loss_w: 结构相似性损失权重 (默认 0.6)
            - shp_loss_w: 形状差异损失权重 (默认 0.4)
            - mean_loss_w: 均值对齐损失权重 (默认 0.2)
            - var_loss_power: 方差损失幂次 (默认 1.5)
            - use_aux: 是否使用辅助分割损失 (默认 False)
    返回:
        dict: 包含以下键的损失配置字典:
            - name: 损失项名称列表
            - op: 损失计算类实例列表
            - weight: 损失权重列表
            - data_src: 数据来源元组列表，格式为 (预测张量名, 标签张量名)
    """

    # ==================== 基础损失配置 ====================
    if cfg.dataset == 'CurveLanes':
        # CurveLanes数据集（复杂弯曲车道场景）
        loss_dict = {
            'name': [
                'cls_loss',  # 主分类损失（行方向，二分类：车道/背景）
                'relation_loss',  # 车道线结构关系损失（保持车道拓扑）
                'relation_dis',  # 车道线形状差异损失（几何一致性）
                'cls_loss_col',  # 列方向分类损失
                'cls_ext',  # 扩展分类任务（行方向附加类别）
                'cls_ext_col',  # 扩展分类任务（列方向附加类别）
                'mean_loss_row',  # 行方向位置均值对齐损失
                'mean_loss_col',  # 列方向位置均值对齐损失
                'var_loss_row',  # 行方向位置方差约束损失（正则化）
                'var_loss_col',  # 列方向位置方差约束损失
                'lane_token_seg_loss_row',  # 行方向车道令牌分割损失
                'lane_token_seg_loss_col'  # 列方向车道令牌分割损失
            ],
            'op': [
                # 公式：Focal Loss = -α(1-p_t)^γ log(p_t), α=1, γ=2
                SoftmaxFocalLoss(2, ignore_lb=-1),  # 处理类别不平衡
                ParsingRelationLoss(),  # 结构相似性损失（公式见下文）
                ParsingRelationDis(),  # 形状差异损失（公式见下文）
                SoftmaxFocalLoss(2, ignore_lb=-1),
                # 标准交叉熵：CE = -Σ y_i log(p_i)
                torch.nn.CrossEntropyLoss(),  # 扩展分类（普通交叉熵）
                torch.nn.CrossEntropyLoss(),
                # 均值损失：L_mean = ||μ_pred - μ_gt||₁
                MeanLoss(),  # 预测位置均值对齐（L1损失）
                MeanLoss(),
                # 方差损失：L_var = |σ_pred² - σ_gt²|^power
                VarLoss(cfg.var_loss_power),  # 方差正则化项
                VarLoss(cfg.var_loss_power),
                TokenSegLoss(),  # 令牌分割损失（实例级交叉熵）
                TokenSegLoss()
            ],
            'weight': [
                1.0,  # 主分类基础权重
                cfg.sim_loss_w,  # 结构相似性权重（建议0.5-1.0）
                cfg.shp_loss_w,  # 形状差异权重（建议0.5-1.0）
                1.0,  # 列分类基础权重
                1.0,  # 扩展分类权重
                1.0,
                cfg.mean_loss_w,  # 均值对齐权重（建议0.1-0.5）
                cfg.mean_loss_w,
                0.01,  # 方差损失权重（小权重正则化）
                0.01,
                1.0,  # 分割损失基础权重
                1.0
            ],
            'data_src': [
                # 格式: (模型输出名, 标签名)
                ('cls_out', 'cls_label'),  # 主分类：二分类输出 vs 标签
                ('cls_out',),  # 结构损失仅需模型输出
                ('cls_out',),  # 形状损失同上
                ('cls_out_col', 'cls_label_col'),  # 列分类
                ('cls_out_ext', 'cls_out_ext_label'),  # 扩展分类（行）
                ('cls_out_col_ext', 'cls_out_col_ext_label'),  # 扩展分类（列）
                ('cls_out', 'cls_label'),  # 行均值：主输出与标签
                ('cls_out_col', 'cls_label_col'),  # 列均值：列输出与标签
                ('cls_out', 'cls_label'),  # 行方差
                ('cls_out_col', 'cls_label_col'),  # 列方差
                ('seg_out_row', 'seg_label'),  # 行分割输出
                ('seg_out_col', 'seg_label')  # 列分割输出
            ],
        }

    elif cfg.dataset in ['Tusimple', 'CULane']:
        # Tusimple/CULane数据集（简单车道场景）
        loss_dict = {
            'name': [
                'cls_loss',  # 主分类损失
                'relation_loss',  # 结构关系损失
                'relation_dis',  # 形状差异损失
                'cls_loss_col',  # 列分类
                'cls_ext',  # 扩展分类（行）
                'cls_ext_col',  # 扩展分类（列）
                'mean_loss_row',  # 行均值对齐
                'mean_loss_col'  # 列均值对齐
            ],
            'op': [
                SoftmaxFocalLoss(2, ignore_lb=-1),
                ParsingRelationLoss(),  # 结构相似性损失
                ParsingRelationDis(),  # 形状差异损失
                SoftmaxFocalLoss(2, ignore_lb=-1),
                torch.nn.CrossEntropyLoss(),
                torch.nn.CrossEntropyLoss(),
                MeanLoss(),  # L1损失：|μ_pred - μ_gt|
                MeanLoss()
            ],
            'weight': [
                1.0,  # 主分类基础权重
                cfg.sim_loss_w,  # 结构相似性权重（0.6典型值）
                cfg.shp_loss_w,  # 形状差异权重（0.4典型值）
                1.0,  # 列分类基础权重
                1.0,  # 扩展分类基础权重
                1.0,
                cfg.mean_loss_w,  # 均值对齐权重（0.2典型值）
                cfg.mean_loss_w
            ],
            'data_src': [
                ('cls_out', 'cls_label'),
                ('cls_out',),  # 结构损失使用主分类输出
                ('cls_out',),
                ('cls_out_col', 'cls_label_col'),
                ('cls_out_ext', 'cls_out_ext_label'),
                ('cls_out_col_ext', 'cls_out_col_ext_label'),
                ('cls_out', 'cls_label'),  # 行均值对齐
                ('cls_out_col', 'cls_label_col')  # 列均值对齐
            ],
        }
    else:
        raise NotImplementedError(f"Unsupported dataset: {cfg.dataset}")

    # ==================== 辅助损失配置 ====================
    if cfg.use_aux:
        # 添加辅助分割损失（增强特征学习）
        loss_dict['name'].append('seg_loss')
        # 类别权重：[背景, 车道1, 车道2, 车道3, 车道4]
        class_weights = torch.tensor([0.6, 1., 1., 1., 1.]).cuda()
        # 交叉熵公式：L = -Σ w_i y_i log(p_i)
        loss_dict['op'].append(torch.nn.CrossEntropyLoss(weight=class_weights).cuda())
        loss_dict['weight'].append(1.0)  # 辅助损失权重
        loss_dict['data_src'].append(('seg_out', 'seg_label'))  # 分割输出与标签

    # 校验配置完整性
    assert len(loss_dict['name']) == len(loss_dict['op']) == \
           len(loss_dict['data_src']) == len(loss_dict['weight'])

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

        