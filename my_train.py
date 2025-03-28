# 导入核心库（移除分布式依赖）
import torch, os, datetime
from tqdm import tqdm  # 替换分布式进度条
from torch.utils.tensorboard import SummaryWriter

# 导入自定义工具模块（简化版本）
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics
from utils.common import (calc_loss, get_model, get_train_loader,
                          inference, merge_config, save_model)
from evaluation.eval_wrapper import eval_lane


def train(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, dataset):
    """单GPU训练函数"""
    net.train()
    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}")  # 普通进度条

    for b_idx, data_label in enumerate(progress_bar):
        global_step = epoch * len(data_loader) + b_idx

        # 前向推理与损失计算
        results = inference(net, data_label, dataset)
        loss = calc_loss(loss_dict, results, logger, global_step, epoch)

        # 反向传播优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)

        # 指标记录（每20步）
        if global_step % 20 == 0:
            reset_metrics(metric_dict)
            update_metrics(metric_dict, results)

            # 记录TensorBoard指标
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar(f'metric/{me_name}', me_op.get(), global_step)
            logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step)

            # 更新进度条
            progress_bar.set_postfix(loss=loss.item())


if __name__ == "__main__":
    # 启用CUDA加速
    torch.backends.cudnn.benchmark = True

    # 解析配置
    args, cfg = merge_config()

    # 工作目录与日志（单GPU简化版）
    work_dir = os.path.join('work_dirs', datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(work_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=os.path.join(work_dir, 'logs'))
    cfg.test_work_dir = work_dir

    # ------------------------ 数据加载 ------------------------
    # 根据网页3建议优化混合锚点数据管道
    train_loader = get_train_loader(cfg)

    # ------------------------ 模型初始化 ------------------------
    net = get_model(cfg).cuda()  # 单GPU直接加载

    # 微调/恢复训练处理（保持与原文一致）
    if cfg.finetune:
        state_dict = torch.load(cfg.finetune)['model']
        net.load_state_dict({k.replace('module.', ''): v for k, v in state_dict.items()}, strict=False)

    if cfg.resume:
        checkpoint = torch.load(cfg.resume)
        net.load_state_dict(checkpoint['model'])
        print(f"Resumed from epoch {checkpoint['epoch']}")

    # ------------------------ 优化器设置 ------------------------
    optimizer = get_optimizer(net, cfg)
    scheduler = get_scheduler(optimizer, cfg, len(train_loader))

    # ------------------------ 训练循环 ------------------------
    best_metric = 0.0
    for epoch in range(cfg.epoch):
        # 训练阶段
        train(net, train_loader, get_loss_dict(cfg), optimizer, scheduler, logger, epoch, get_metric_dict(cfg),
              cfg.dataset)

        # 评估与保存（根据网页1的混合锚点验证逻辑）
        current_metric = eval_lane(net, cfg, ep=epoch, logger=logger)
        if current_metric > best_metric:
            best_metric = current_metric
            save_model(net, optimizer, epoch, work_dir, distributed=False)
            print(f"New best model saved at epoch {epoch} with metric {best_metric:.4f}")

    logger.close()