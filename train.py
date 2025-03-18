# 导入核心库
import torch, os, datetime

# 导入自定义工具模块
from utils.dist_utils import dist_print, dist_tqdm, synchronize  # 分布式训练工具
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler  # 工厂方法
from utils.metrics import update_metrics, reset_metrics  # 指标计算
from utils.common import (calc_loss, get_model, get_train_loader,
                          inference, merge_config, save_model, cp_projects)  # 通用工具
from utils.common import get_work_dir, get_logger  # 工作目录和日志记录
import time
from evaluation.eval_wrapper import eval_lane  # 评估模块


def train(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, dataset):
    """训练函数"""
    net.train()
    progress_bar = dist_tqdm(train_loader)  # 分布式进度条

    for b_idx, data_label in enumerate(progress_bar):
        global_step = epoch * len(data_loader) + b_idx  # 计算全局步数

        # 前向推理
        results = inference(net, data_label, dataset)

        # 计算损失
        loss = calc_loss(loss_dict, results, logger, global_step, epoch)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)  # 更新学习率

        # 每隔20步记录指标
        if global_step % 20 == 0:
            reset_metrics(metric_dict)
            update_metrics(metric_dict, results)

            # 记录指标到TensorBoard
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)
            logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            # 更新进度条显示
            if hasattr(progress_bar, 'set_postfix'):
                kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in
                          zip(metric_dict['name'], metric_dict['op'])}
                new_kwargs = {}
                for k, v in kwargs.items():
                    if 'lane' in k:  # 过滤车道线相关指标
                        continue
                    new_kwargs[k] = v
                progress_bar.set_postfix(loss='%.3f' % float(loss), **new_kwargs)


if __name__ == "__main__":
    # 启用CUDA加速
    torch.backends.cudnn.benchmark = True

    # 解析命令行参数和配置文件
    args, cfg = merge_config()

    # ------------------------ 分布式训练设置 ------------------------
    if args.local_rank == 0:
        work_dir = get_work_dir(cfg)  # 创建工作目录

    # 检测是否分布式环境
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1

    if distributed:
        # 初始化分布式训练
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        # 同步工作目录路径
        if args.local_rank == 0:
            with open('.work_dir_tmp_file.txt', 'w') as f:
                f.write(work_dir)
        else:
            while not os.path.exists('.work_dir_tmp_file.txt'):
                time.sleep(0.1)
            with open('.work_dir_tmp_file.txt', 'r') as f:
                work_dir = f.read().strip()

    synchronize()  # 进程同步
    cfg.test_work_dir = work_dir
    cfg.distributed = distributed

    # ------------------------ 初始化阶段 ------------------------
    if args.local_rank == 0:
        os.system('rm .work_dir_tmp_file.txt')

    # 打印启动信息
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    dist_print(cfg)
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide', '34fca']

    # ------------------------ 数据加载 ------------------------
    train_loader = get_train_loader(cfg)  # 获取训练数据加载器

    # ------------------------ 模型初始化 ------------------------
    net = get_model(cfg)  # 根据配置创建模型
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    # ------------------------ 优化器设置 ------------------------
    optimizer = get_optimizer(net, cfg)  # 根据配置创建优化器

    # 微调/恢复训练处理
    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # 仅加载主干网络参数
        for k, v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)

    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])  # 恢复优化器状态
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1  # 计算恢复的epoch
    else:
        resume_epoch = 0

    # ------------------------ 训练准备 ------------------------
    scheduler = get_scheduler(optimizer, cfg, len(train_loader))  # 学习率调度器
    metric_dict = get_metric_dict(cfg)  # 获取评估指标
    loss_dict = get_loss_dict(cfg)  # 获取损失函数
    logger = get_logger(work_dir, cfg)  # 初始化日志记录器

    # ------------------------ 主训练循环 ------------------------
    max_res = 0  # 记录最佳评估结果
    res = None
    for epoch in range(resume_epoch, cfg.epoch):
        # 训练阶段
        train(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, cfg.dataset)
        train_loader.reset()  # 重置数据加载器（如有必要）

        # 评估阶段
        res = eval_lane(net, cfg, ep=epoch, logger=logger)

        # 保存最佳模型
        if res is not None and res > max_res:
            max_res = res
            save_model(net, optimizer, epoch, work_dir, distributed)

        # 记录评估结果
        logger.add_scalar('CuEval/X', max_res, global_step=epoch)

    # 关闭日志记录器
    logger.close()
