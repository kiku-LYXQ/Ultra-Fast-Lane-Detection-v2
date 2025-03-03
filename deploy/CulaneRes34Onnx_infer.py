"""
基于ONNX Runtime的Ultra-Fast-Lane-Detection-v2推理实现
功能：实现车道线检测的预处理、模型推理和后处理全流程
特点：支持动态输入尺寸，保留与原始TensorRT版本相同的处理逻辑
"""

import cv2
import numpy as np
import onnxruntime as ort  # ONNX Runtime推理库
import torch
import argparse
import os
import sys

# 添加项目根目录到系统路径，确保模块导入正确
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.config import Config  # 配置文件加载类


class UFLDv2_ONNX:
    def __init__(self, onnx_path, config_path, ori_size):
        """
        初始化车道线检测器
        参数：
            onnx_path   : ONNX模型文件路径
            config_path : 训练配置文件路径
            ori_size    : 原始输入图像尺寸（宽, 高）
        """
        # 初始化ONNX Runtime推理会话
        # 使用默认提供器（优先使用GPU如果可用）
        self.session = ort.InferenceSession(onnx_path)

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
        loc_row = torch.from_numpy(pred[0])  # 行位置预测（形状：[1, num_row, num_cls_row, 4]）
        loc_col = torch.from_numpy(pred[1])  # 列位置预测（形状：[1, num_col, num_cls_col, 4]）
        exist_row = torch.from_numpy(pred[2])  # 行存在性预测
        exist_col = torch.from_numpy(pred[3])  # 列存在性预测

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
        完整处理流程：预处理 → 推理 → 后处理 → 可视化
        参数：
            img : 输入图像（BGR格式，numpy数组）
        """
        im0 = img.copy()  # 保留原始图像用于可视化

        # 图像预处理 ------------------------------------------------------------
        # 1. 裁剪顶部区域（根据训练时的裁剪比例）
        img = img[self.cut_height:, :, :]
        # 2. 调整到模型输入尺寸（双三次插值保持清晰度）
        img = cv2.resize(img, (self.input_width, self.input_height), cv2.INTER_CUBIC)
        # 3. 归一化到[0,1]范围
        img = img.astype(np.float32) / 255.0
        # 4. 调整维度顺序为NCHW（模型输入要求）
        # 原始形状：HWC (height, width, channel)
        # 转换后：NCHW (batch=1, channel, height, width)
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]

        # ONNX推理 -------------------------------------------------------------
        outputs = self.session.run(
            self.output_names,  # 需要获取的输出节点名称列表
            {self.input_name: img}  # 输入数据字典
        )
        # 输出顺序对应：loc_row, loc_col, exist_row, exist_col

        # 后处理：转换坐标 ------------------------------------------------------
        coords = self.pred2coords(outputs)

        # 可视化结果 -----------------------------------------------------------
        # 在原始图像上绘制检测到的车道线点
        for lane in coords:  # 遍历每条车道线
            for coord in lane:  # 遍历车道线的每个点
                # 绘制绿色实心圆点（半径2px）
                cv2.circle(im0, coord, 2, (0, 255, 0), -1)
        cv2.imshow("result", im0)  # 显示结果


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
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cap = cv2.VideoCapture(args.video_path)  # 打开视频文件
    # 初始化检测器
    detector = UFLDv2_ONNX(args.onnx_path, args.config_path, args.ori_size)

    while True:
        success, img = cap.read()
        if not success:  # 视频读取结束
            break

        # 视频帧预处理 ----------------------------------------------------------
        # 1. 调整到1600x903分辨率（适配原始模型训练尺寸）
        img = cv2.resize(img, (1600, 903))
        # 2. 裁剪ROI区域（380:700行，所有列）
        img = img[380:700, :, :]

        # 执行检测
        detector.forward(img)

        # 按Q退出
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
