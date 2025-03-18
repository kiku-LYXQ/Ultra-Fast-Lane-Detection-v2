# 数据集配置
dataset = 'Tusimple'       # 使用Tusimple车道线数据集
data_root = ''              # 数据集根目录路径（需根据实际路径修改）
crop_ratio = 0.8           # 图像随机裁剪比例（保留原图的80%区域）

# 训练超参数
epoch = 100                 # 总训练轮数
batch_size = 32             # 每个批次的样本数量
optimizer = 'SGD'           # 优化器类型（SGD/Adam等）
learning_rate = 0.05        # 初始学习率
weight_decay = 0.0001       # L2正则化系数
momentum = 0.9              # SGD动量参数

# 学习率调度策略
scheduler = 'multi'          # 多阶段学习率调整策略
steps = [50, 75]            # 在第50和75轮降低学习率
gamma = 0.1                 # 学习率衰减系数（每次衰减为原来的0.1倍）
warmup = 'linear'           # 学习率热身策略（线性增加）
warmup_iters = 100         # 热身阶段迭代次数（前100次迭代逐步提升学习率）

# 模型架构
backbone = '18'             # 主干网络（ResNet-18）
num_lanes = 4               # 最多检测4条车道线
use_aux = False             # 是否使用辅助分支（如分割分支）
fc_norm = False             # 全连接层是否添加归一化

# 空间量化参数
griding_num = 100           # 横向网格划分数量（将图像宽度分为100个单元）
num_row = 56                # 纵向锚点数量（Y轴方向）
num_col = 41                # 横向锚点数量（X轴方向）
train_width = 800           # 训练时图像输入宽度（像素）
train_height = 320          # 训练时图像输入高度（像素）
num_cell_row = 100          # 纵向网格单元数（用于特征图量化）
num_cell_col = 100          # 横向网格单元数

# 损失函数权重
sim_loss_w = 0.0            # 相似性损失权重（未启用）
shp_loss_w = 0.0            # 形状约束损失权重（未启用）
mean_loss_w = 0.05          # 均值约束损失权重
mean_loss_col_w = 0.05      # 列方向均值约束损失权重
cls_loss_col_w = 1.0         # 列分类损失基础权重
cls_ext_col_w = 1.0          # 列分类扩展权重
var_loss_power = 2.0         # 方差损失指数（L2损失）
soft_loss = True            # 是否使用软分类（概率分布）

# 系统配置
log_path = ''               # 训练日志保存路径
finetune = None             # 微调时加载的预训练模型路径
resume = None               # 恢复训练时加载的检查点路径
test_model = ''             # 测试模型文件路径
test_work_dir = ''          # 测试结果输出目录
auto_backup = True          # 是否自动备份配置文件

# 评估模式
eval_mode = 'normal'        # 评估模式（normal/extreme等不同场景）