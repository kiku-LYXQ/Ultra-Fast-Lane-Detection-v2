"""
基于ONNX Runtime的Ultra-Fast-Lane-Detection-v2推理实现
功能：实现车道线检测的预处理、模型推理和后处理全流程
特点：支持动态输入尺寸，保留与原始TensorRT版本相同的处理逻辑
"""

import cv2
import numpy as np
import onnxruntime as ort  # ONNX Runtime推理库
import torch   # 引入了torch所以不会有cudnn链接库报错这种问题
import argparse
import os
import sys

# 添加项目根目录到系统路径，确保模块导入正确
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.config import Config  # 配置文件加载类


class UFLDv2_ONNX:
    def __init__(self, onnx_path, config_path, ori_size, use_gpu=False):
        """
        初始化车道线检测器
        参数：
            onnx_path   : ONNX模型文件路径
            config_path : 训练配置文件路径
            ori_size    : 原始输入图像尺寸（宽, 高）
        """
        # 初始化ONNX Runtime推理会话
        # 配置执行提供器优先级
        providers = ['CPUExecutionProvider']  # 默认使用CPU
        if use_gpu:
            # 优先尝试CUDA，然后是TensorRT（如果可用）
            providers = [
                ('CUDAExecutionProvider', {
                    'device_id': 0,
                    'arena_extend_strategy': 'kNextPowerOfTwo',
                    'gpu_mem_limit': 1 * 1024 * 1024 * 1024,  # 限制1GB显存
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
        print("可用Providers:", ort.get_available_providers())
        print(f"当前使用的执行提供器：{self.session.get_providers()}")
        if use_gpu and 'CUDAExecutionProvider' in self.session.get_providers():
            print("CUDA GPU加速已启用")
        else:
            print("使用CPU进行推理")

        # 获取输入输出节点信息 -----------------------------------------------------
        # 输入节点名称（模型只有一个输入）
        self.input_name = self.session.get_inputs()[0].name
        # 输出节点名称列表（loc_row, loc_col, exist_row, exist_col）
        self.output_names = [output.name for output in self.session.get_outputs()]

        # 加载配置文件参数 --------------------------------------------------------
        cfg = Config.fromfile(config_path)
        # 原始图像尺寸（用于坐标映射）
        self.ori_img_w, self.ori_img_h = ori_size
        # 裁剪高度（根据训练配置计算）
        self.cut_height = int(cfg.train_height * (1 - cfg.crop_ratio))
        # 模型输入尺寸（需要与训练时保持一致）
        self.input_width = cfg.train_width
        self.input_height = cfg.train_height
        # 行/列锚点数量（关键参数，影响检测精度）
        self.num_row = cfg.num_row
        self.num_col = cfg.num_col

        # 生成行/列锚点坐标（归一化后的相对位置）
        # 行锚点在垂直方向均匀分布（从图像高度42%处开始）
        self.row_anchor = np.linspace(0.42, 1, self.num_row)
        # 列锚点在水平方向均匀分布（覆盖整个图像宽度）
        self.col_anchor = np.linspace(0, 1, self.num_col)

    def pred2coords(self, pred):
        """
        将模型输出转换为实际车道线坐标
        参数：
            pred : 模型输出元组，包含四个输出张量的numpy数组
        返回：
            coords : 车道线坐标列表，每个元素是该车道线的坐标点列表
        """
        # 将numpy数组转换为torch张量（保持与原始实现兼容）
        loc_row = torch.from_numpy(pred[0])  # 行位置预测（形状：[1, num_grid_row, num_cls_row, 4]）
        loc_col = torch.from_numpy(pred[1])  # 列位置预测（形状：[1, num_grid_row, num_cls_col, 4]）
        exist_row = torch.from_numpy(pred[2])  # 行存在性预测 [1, num_grid_row, 2, 4]
        exist_col = torch.from_numpy(pred[3])  # 列存在性预测 [1, num_grid_row, 2, 4]

        # 获取各维度尺寸
        batch_size, num_grid_row, num_cls_row, num_lane_row = loc_row.shape
        batch_size, num_grid_col, num_cls_col, num_lane_col = loc_col.shape

        # 获取最大概率索引（确定最可能的位置） ----------------------------------------
        # 行位置的最大索引（形状：[1, num_cls_row, 4]）
        max_indices_row = loc_row.argmax(1)
        # 行存在性的最大索引（0表示不存在，1表示存在）
        valid_row = exist_row.argmax(1)
        # 列位置的最大索引
        max_indices_col = loc_col.argmax(1)
        # 列存在性的最大索引
        valid_col = exist_col.argmax(1)

        coords = []  # 存储所有车道线的坐标
        row_lane_idx = [1, 2]  # 行车道线索引（根据数据集定义）
        col_lane_idx = [0, 3]  # 列车道线索引

        # 处理行车道线（垂直方向车道线）---------------------------------------------
        for i in row_lane_idx:
            tmp = []
            # 存在性判断：超过半数的锚点认为存在车道线
            if valid_row[0, :, i].sum() > num_cls_row / 2:
                # 遍历每个行锚点
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:  # 当前锚点存在车道线
                        # 在预测位置周围创建索引窗口（提高位置精度）
                        all_ind = torch.tensor(list(range(
                            max(0, max_indices_row[0, k, i] - self.input_width),
                            min(num_grid_row - 1, max_indices_row[0, k, i] + self.input_width) + 1
                        )))

                        # 计算加权平均位置 -------------------------------------------------
                        # 使用softmax计算窗口内各位置的权重
                        # 公式：out_tmp = Σ(softmax(loc_row[all_ind]) * all_ind) + 0.5
                        out_tmp = (loc_row[0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                        # 将归一化位置映射回原图宽度
                        out_tmp = out_tmp / (num_grid_row - 1) * self.ori_img_w
                        # 计算对应y坐标（行锚点位置）
                        y_coord = int(self.row_anchor[k] * self.ori_img_h)
                        tmp.append((int(out_tmp), y_coord))
                coords.append(tmp)

        # 处理列车道线（水平方向车道线）---------------------------------------------
        for i in col_lane_idx:
            tmp = []
            # 存在性判断：超过1/4的锚点认为存在
            if valid_col[0, :, i].sum() > num_cls_col / 4:
                for k in range(valid_col.shape[1]):
                    if valid_col[0, k, i]:
                        all_ind = torch.tensor(list(range(
                            max(0, max_indices_col[0, k, i] - self.input_width),
                            min(num_grid_col - 1, max_indices_col[0, k, i] + self.input_width) + 1
                        )))

                        # 计算加权平均位置（与行处理类似）
                        out_tmp = (loc_col[0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                        # 映射回原图高度
                        out_tmp = out_tmp / (num_grid_col - 1) * self.ori_img_h
                        # 计算对应x坐标（列锚点位置）
                        x_coord = int(self.col_anchor[k] * self.ori_img_w)
                        tmp.append((x_coord, int(out_tmp)))
                coords.append(tmp)
        return coords

    def forward(self, img):
        """
        完整的车道线检测流程：预处理 → 推理 → 后处理 → 可视化
        参数：
            img : 输入图像（BGR格式，numpy数组，形状为[H, W, 3]）
        返回：
            coords : 处理后的车道线坐标列表
        """
        # ============================ 视频帧预处理 ==============================
        # 阶段目标：将输入帧适配到模型训练时的视野范围和比例

        # 1. 基准尺寸调整（双线性插值平衡效率与质量）
        resized_frame = cv2.resize(img, (1600, 903), interpolation=cv2.INTER_LINEAR)

        # 2. 关键区域提取（380:700行，去除天空和引擎盖干扰）
        roi_frame = resized_frame[380:700, :, :]  # 获取高度320px的ROI

        # 3. 保留原始ROI用于可视化（避免后续处理影响显示）
        visual_frame = roi_frame.copy()  # 类型：numpy.ndarray (320, 1600, 3)

        # ============================ 模型输入预处理 =============================
        # 阶段目标：生成符合模型输入规范的张量，需与训练预处理严格一致

        # 1. 裁剪顶部区域（根据训练配置去除近场视野）
        model_input = roi_frame[self.cut_height:, :, :]  # 形状变为 (320-cut_height, 1600, 3)

        # 2. 尺寸标准化（双三次插值保留几何特征）
        model_input = cv2.resize(
            model_input,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_CUBIC
        )  # 输出形状 (input_height, input_width, 3)

        # 3. 归一化处理（匹配训练时的数据标准化方式）
        model_input = model_input.astype(np.float32) / 255.0

        # 4. 转换为NCHW格式（添加批次维度，调整通道顺序）
        input_tensor = np.transpose(model_input, (2, 0, 1))[np.newaxis, ...]  # 形状 (1, 3, H, W)

        # 5. 确保内存连续性（提升推理效率）
        input_tensor = np.ascontiguousarray(input_tensor)

        # ============================== 模型推理 ================================
        # 使用ONNX Runtime进行高效推理
        model_outputs = self.session.run(
            output_names=self.output_names,  # 预加载的输出节点名称
            input_feed={self.input_name: input_tensor}  # 输入数据字典
        )  # 输出顺序：loc_row, loc_col, exist_row, exist_col

        # ============================ 后处理流程 ================================
        # 将模型输出转换为实际坐标
        coords = self.pred2coords(model_outputs)  # 坐标格式 [[(x1,y1), (x2,y2), ...], ...]

        # ============================= 结果可视化 ================================
        # 在原始ROI图像上叠加检测结果
        # 仅绘制检测点
        for lane_points in coords:  # 遍历每条车道线
            if len(lane_points) == 0:
                continue
            # 绘制每个检测点
            for pt in lane_points:
                # 将浮点坐标转换为整数像素位置
                x, y = int(pt[0]), int(pt[1])
                # 绘制绿色实心圆点（半径3px）
                cv2.circle(visual_frame, (x, y), radius=3, color=(0, 255, 0), thickness=-1)

        # 实时显示检测结果
        cv2.imshow("Lane Detection", visual_frame)
        cv2.waitKey(1)  # 允许图像窗口更新

        return coords


def get_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/culane_res34.py',
                        help='模型配置文件路径', type=str)
    parser.add_argument('--onnx_path', default='weights/culane_res34.onnx',
                        help='ONNX模型路径', type=str)
    parser.add_argument('--video_path', default='example.mp4',
                        help='输入视频路径', type=str)
    parser.add_argument('--ori_size', default=(1600, 320),
                        help='原始图像尺寸（宽, 高）', type=tuple)
    parser.add_argument('--use_gpu', action='store_true',
                        help='启用CUDA GPU加速（需要安装onnxruntime-gpu）')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cap = cv2.VideoCapture(args.video_path)  # 打开视频文件
    # 初始化检测器
    detector = UFLDv2_ONNX(args.onnx_path, args.config_path, args.ori_size, use_gpu = args.use_gpu)

    while True:
        success, img = cap.read()
        if not success:  # 视频读取结束
            break

        # 执行检测
        detector.forward(img)

        # 按Q退出
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
