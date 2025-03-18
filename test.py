# 导入必要的库和模块
import torch, os
from utils.common import merge_config, get_model  # 从utils模块导入配置合并和模型构建函数
from evaluation.eval_wrapper import eval_lane  # 导入评估车道检测模型的函数

# 主程序入口
if __name__ == "__main__":
    # 启用cudnn的自动优化内核，提升计算效率（适用于输入尺寸固定的场景）
    torch.backends.cudnn.benchmark = True

    # 合并命令行参数和配置文件，获取配置对象args和cfg
    args, cfg = merge_config()

    # 分布式训练初始化
    distributed = False
    # 检查环境变量中是否存在WORLD_SIZE，判断是否处于分布式环境
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1  # WORLD_SIZE表示进程总数
    cfg.distributed = distributed  # 将分布式配置保存到cfg对象中

    # 如果处于分布式环境，初始化进程组
    if distributed:
        # 设置当前进程使用的GPU设备（根据local_rank参数）
        torch.cuda.set_device(args.local_rank)
        # 初始化分布式进程组，使用nccl后端（NVIDIA GPU推荐）
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # 构建模型：根据配置获取模型实例
    net = get_model(cfg)

    # 加载预训练模型权重
    # 从指定路径加载模型状态字典，并映射到CPU内存（避免GPU内存不足）
    state_dict = torch.load(cfg.test_model, map_location='cpu')['model']
    compatible_state_dict = {}
    # 遍历状态字典的键值对，处理多GPU训练保存的模型前缀
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v  # 去除键名中的'module.'前缀（多GPU情况下的模型保存）
        else:
            compatible_state_dict[k] = v  # 保留原始键名

    # 将处理后的状态字典加载到模型中，strict=True确保严格匹配所有键
    net.load_state_dict(compatible_state_dict, strict=True)

    # 如果处于分布式环境，将模型包裹在DistributedDataParallel中，实现数据并行
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    # 创建测试输出目录（如果不存在）
    if not os.path.exists(cfg.test_work_dir):
        os.mkdir(cfg.test_work_dir)

    # 执行车道线评估：调用评估函数，传入模型和配置参数
    eval_lane(net, cfg)