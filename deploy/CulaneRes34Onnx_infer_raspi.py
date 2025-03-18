"""
基于ONNX Runtime的Ultra-Fast-Lane-Detection-v2推理实现
核心算法解析：
1. 行/列锚点机制：在预设的锚点位置进行密集预测，大幅减少计算量
2. 多分类预测：每个锚点位置预测多个可能的位置偏移量
3. 动态加权融合：通过softmax加权融合多个预测结果，提高位置精度
"""

import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os
import sys

# 添加项目根目录到系统路径，确保模块导入正确
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.config import Config  # 配置文件加载类

# 自定义softmax函数（兼容PyTorch的维度行为）
def softmax(x, axis=-1):
    """
    实现数值稳定的softmax计算
    参数：
        x    : 输入数据（numpy数组）
        axis : 计算softmax的维度
    返回：
        softmax计算结果（概率分布）
    """
    # 减去最大值防止指数爆炸（数值稳定性处理）
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / e_x.sum(axis=axis, keepdims=True)

class UFLDv2_ONNX:
    def __init__(self, onnx_path, config_path, ori_size, use_gpu=False):
        """
        初始化车道线检测器（关键参数说明）
        参数详解：
            onnx_path   : ONNX模型文件路径（需与训练时的输入输出对齐）
            config_path : 训练配置文件路径（包含模型参数和训练配置）
            ori_size    : 原始输入图像尺寸（宽, 高）（影响坐标映射精度）
            use_gpu     : 是否启用CUDA GPU加速
        """
        # ONNX Runtime会话初始化 ----------------------------------------------
        # 使用默认执行提供器（CPUExecutionProvider）
        # 注意：树莓派上需要安装onnxruntime的ARM兼容版本
        # 配置执行提供器优先级
        providers = ['CPUExecutionProvider']  # 默认使用CPU
        if use_gpu:
            # 优先尝试CUDA，然后是TensorRT（如果可用）
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 2 * 1024 * 1024 * 1024,  # 限制2GB显存
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'do_copy_in_default_stream': True,
                }),
                'CPUExecutionProvider'
            ]

        # 初始化ONNX Runtime会话（添加providers参数）
        self.session = ort.InferenceSession(
            onnx_path,
            providers=providers  # 指定执行提供器优先级
        )

        # 检查GPU是否实际启用（调试信息）
        print(f"当前使用的执行提供器：{self.session.get_providers()}")
        if use_gpu and 'CUDAExecutionProvider' in self.session.get_providers():
            print("CUDA GPU加速已启用")
        else:
            print("使用CPU进行推理")

        # 输入输出节点信息获取（模型部署关键信息）-------------------------------
        # 输入节点名称（通常为单个输入）
        self.input_name = self.session.get_inputs()[0].name
        # 输出节点名称列表（顺序需与训练时保持一致）
        self.output_names = [output.name for output in self.session.get_outputs()]

        # 配置文件参数解析（关键模型参数）---------------------------------------
        # 注意：Config类需要实现fromfile方法解析配置文件
        cfg = Config.fromfile(config_path)

        # 原始图像尺寸（用于将归一化坐标映射回实际图像）
        self.ori_img_w, self.ori_img_h = ori_size

        # 裁剪高度计算（根据训练时的裁剪比例）
        # 公式：cut_height = 训练高度 × (1 - 裁剪比例)
        self.cut_height = int(cfg.train_height * (1 - cfg.crop_ratio))

        # 模型输入尺寸（必须与训练时完全一致）
        self.input_width = cfg.train_width    # 通常为800
        self.input_height = cfg.train_height  # 通常为320

        # 锚点数量配置（直接影响检测精度和速度）
        self.num_row = cfg.num_row  # 行锚点数（垂直方向采样点，通常72个）
        self.num_col = cfg.num_col  # 列锚点数（水平方向采样点，通常81个）

        # 生成归一化锚点坐标（核心检测位置）-------------------------------------
        # 行锚点：在垂直方向0.42到1.0之间均匀分布（关注图像下半部分）
        self.row_anchor = np.linspace(0.42, 1, self.num_row)
        # 列锚点：在水平方向0.0到1.0之间均匀分布（覆盖整个图像宽度）
        self.col_anchor = np.linspace(0, 1, self.num_col)

    def pred2coords(self, pred):
        """
        核心后处理函数：将模型输出转换为实际车道线坐标
        处理流程：
        1. 解析模型输出的四个张量
        2. 通过argmax获取最大概率位置
        3. 使用滑动窗口加权平均提高定位精度
        4. 映射到原始图像坐标系
        """
        # 模型输出解析（顺序必须与模型输出一致）-------------------------------
        loc_row = pred[0]  # 行位置预测 [1, num_row, num_cls_row, 4]
        loc_col = pred[1]  # 列位置预测 [1, num_col, num_cls_col, 4]
        exist_row = pred[2]  # 行存在性概率 [1, num_cls_row, 4]
        exist_col = pred[3]  # 列存在性概率 [1, num_cls_col, 4]

        # 最大概率索引获取（numpy实现）----------------------------------------
        # loc_row形状说明：[batch, num_grid_row, num_cls_row, num_lane_row]
        max_indices_row = np.argmax(loc_row, axis=1)  # 沿num_grid_row维度取最大值
        valid_row = np.argmax(exist_row, axis=1)      # 存在性判断（0/1）

        max_indices_col = np.argmax(loc_col, axis=1)
        valid_col = np.argmax(exist_col, axis=1)

        coords = []  # 最终坐标容器
        row_lane_idx = [1, 2]  # 行车道线索引（根据CULane数据集定义）
        col_lane_idx = [0, 3]  # 列车道线索引

        # 行车道线处理（垂直方向）---------------------------------------------
        for i in row_lane_idx:
            tmp = []
            # 存在性判断：有效锚点超过半数则视为存在车道线
            if valid_row[0, :, i].sum() > loc_row.shape[2] / 2:
                # 遍历每个行锚点（垂直方向采样点）
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:  # 当前锚点存在车道线
                        # 生成索引窗口（在预测位置周围扩展）
                        current_max = max_indices_row[0, k, i]
                        start = max(0, current_max - self.input_width)
                        end = min(loc_row.shape[1]-1, current_max + self.input_width) + 1
                        all_ind = np.arange(start, end)

                        # 加权平均计算（关键精度提升步骤）-----------------------
                        # 公式：out_tmp = Σ(softmax(score) * index) + 0.5
                        weights = softmax(loc_row[0, all_ind, k, i], axis=0)
                        out_tmp = np.sum(weights * all_ind) + 0.5  # +0.5用于四舍五入

                        # 坐标映射：将归一化位置转换为实际像素坐标
                        # 映射公式：实际坐标 = 预测位置 / (网格数-1) × 原图宽度
                        out_tmp = out_tmp / (loc_row.shape[1]-1) * self.ori_img_w

                        # 计算对应y坐标（基于行锚点位置）
                        y_coord = int(self.row_anchor[k] * self.ori_img_h)
                        tmp.append((int(out_tmp), y_coord))
                coords.append(tmp)

        # 列车道线处理（水平方向，逻辑类似）-------------------------------------
        for i in col_lane_idx:
            tmp = []
            # 存在性阈值稍低（1/4有效锚点即视为存在）
            if valid_col[0, :, i].sum() > loc_col.shape[2] / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        current_max = max_indices_col[0, k, i]
                        start = max(0, current_max - self.input_width)
                        end = min(loc_col.shape[1]-1, current_max + self.input_width) + 1
                        all_ind = np.arange(start, end)

                        weights = softmax(loc_col[0, all_ind, k, i], axis=0)
                        out_tmp = np.sum(weights * all_ind) + 0.5

                        # 注意此处映射到高度方向
                        out_tmp = out_tmp / (loc_col.shape[1]-1) * self.ori_img_h

                        x_coord = int(self.col_anchor[k] * self.ori_img_w)
                        tmp.append((x_coord, int(out_tmp)))
                coords.append(tmp)
        return coords

    def forward(self, img):
        """
        完整处理流程（关键步骤说明）：
        1. 图像预处理：裁剪→缩放→归一化→维度变换
        2. ONNX推理：单次前向计算
        3. 后处理：坐标转换与存在性判断
        4. 可视化：在原始图像上绘制检测结果
        """
        im0 = img.copy()  # 保留原始图像用于可视化

        # 图像预处理流水线 ----------------------------------------------------
        # 步骤1：顶部裁剪（去除天空等无关区域）
        img = img[self.cut_height:, :, :]

        # 步骤2：双三次插值缩放（保持图像清晰度）
        img = cv2.resize(img, (self.input_width, self.input_height),
                        interpolation=cv2.INTER_CUBIC)

        # 步骤3：归一化到[0,1]范围（与训练数据一致）
        img = img.astype(np.float32) / 255.0

        # 步骤4：维度变换HWC→NCHW（模型输入要求）
        # 说明：添加batch维度，通道优先
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

        # ONNX推理执行 -------------------------------------------------------
        outputs = self.session.run(
            self.output_names,  # 需要获取的输出节点列表
            {self.input_name: img}  # 输入数据字典
        )

        # 后处理与可视化 ------------------------------------------------------
        coords = self.pred2coords(outputs)

        # 可视化逻辑：在原始图像绘制绿色圆点
        for lane in coords:
            for coord in lane:
                # 使用OpenCV绘制实心圆点（坐标必须为整数）
                cv2.circle(im0, coord, 2, (0, 255, 0), -1)

        cv2.imshow("result", im0)  # 实时显示结果


def get_args():
    """命令行参数解析配置"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/culane_res34.py',
                       help='模型配置文件路径（需包含训练参数）')
    parser.add_argument('--onnx_path', default='weights/culane_res34.onnx',
                       help='导出的ONNX模型路径')
    parser.add_argument('--video_path', default='example.mp4',
                       help='输入视频路径（支持MP4/AVI格式）')
    parser.add_argument('--ori_size', default=(1600, 320), type=tuple,
                       help='原始训练尺寸（宽，高）需与模型匹配')
    parser.add_argument('--use_gpu', action='store_true',
                        help='启用CUDA GPU加速（需要安装onnxruntime-gpu）')
    return parser.parse_args()


if __name__ == "__main__":
    # 程序主入口 --------------------------------------------------------------
    args = get_args()

    # 视频流初始化
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        raise IOError("无法打开视频文件")

    # 检测器初始化
    detector = UFLDv2_ONNX(
        args.onnx_path,
        args.config_path,
        args.ori_size,
        use_gpu = args.use_gpu  # 新增参数
    )

    # 主循环处理帧
    while True:
        success, img = cap.read()
        if not success:
            break

        # 视频帧预处理（适配模型输入尺寸）
        # 步骤1：调整到训练分辨率（1600x903是CULane数据集的常用尺寸）
        img = cv2.resize(img, (1600, 903))

        # 步骤2：裁剪ROI区域（380:700行对应实际道路区域）
        img = img[380:700, :, :]

        # 执行检测
        detector.forward(img)

        # 退出控制
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 资源释放
    cap.release()
    cv2.destroyAllWindows()
