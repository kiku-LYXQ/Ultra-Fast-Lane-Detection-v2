# 数据集配置
dataset = 'CULane'                 # 使用的数据集名称（CULane车道线检测数据集）
data_root = '/home/lxy/lxy_project/clrnet/data/CULane'                     # 【必须修改】数据集根目录路径（存放训练/测试数据的文件夹路径）

# 训练超参数
epoch = 50                         # 总训练轮次（整个数据集遍历次数）
batch_size = 10                    # 每批次数据量（根据GPU显存调整，显存不足需减小） 原始32
optimizer = 'SGD'                  # 优化器类型（可选SGD/Adam等）
learning_rate = 0.05               # 初始学习率（SGD通常使用较大学习率）
weight_decay = 0.0001              # 权重衰减（L2正则化系数，防止过拟合）
momentum = 0.9                     # SGD动量参数（加速收敛，减少震荡）

# 学习率调度
scheduler = 'multi'                # 学习率调整策略（'multi'表示多步衰减）
steps = [25, 38]                   # 在第25和38个epoch降低学习率
gamma = 0.1                        # 学习率衰减系数（每次衰减为之前的0.1倍）
warmup = 'linear'                  # 热身策略（训练初期逐步增加学习率）
warmup_iters = 695                 # 热身阶段迭代次数（避免初始阶段不稳定）

# 模型结构
backbone = '18'                    # 主干网络（ResNet-18，可选18/34/50等）
griding_num = 200                  # 车道线水平方向分格数（影响定位精度）  原始200
use_aux = False                    # 是否使用辅助分支（增加训练稳定性，但增加计算量）
num_row = 72                       # 行锚点数 原始72
num_col = 81                       # 列锚点数   原始81
fc_norm = True                     # 是否在全连接层使用归一化

# 损失函数权重
sim_loss_w = 0.0                   # 结构相似度损失权重（当前未启用）
shp_loss_w = 0.0                   # 形状相似损失权重（当前未启用）
mean_loss_w = 0.05                 # 均值损失权重（平衡预测分布）
var_loss_power = 2.0               # 方差损失幂次（L2损失时为2）

# 数据预处理
train_width = 1600                 # 训练图像宽度（输入网络前的缩放尺寸）
train_height = 320                 # 训练图像高度
crop_ratio = 0.6                   # 随机裁剪比例（数据增强，防过拟合）
num_cell_row = 200                 # 特征图行单元格数（可能用于损失计算）
num_cell_col = 100                 # 特征图列单元格数

# 测试配置
test_model = ''                    # 测试模型路径（.pth文件）
test_work_dir = ''                 # 测试输出目录（保存预测结果）
tta = True                         # 测试时数据增强（提升精度但增加耗时）
num_lanes = 4                      # 最多检测车道线数量（根据数据集设置）

# 系统与日志
auto_backup = True                 # 自动备份代码（防止训练意外中断）
log_path = ''                      # 训练日志保存路径（需指定具体目录）
finetune = None                    # 微调模型路径（从预训练模型继续训练）
resume = None                      # 恢复训练检查点路径（包含优化器状态）
note = ''                          # 备注信息（记录实验配置变更）