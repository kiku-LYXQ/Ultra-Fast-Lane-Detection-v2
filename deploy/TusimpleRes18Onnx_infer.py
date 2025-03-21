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
        self.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
        self.col_anchor = np.linspace(0, 1, cfg.num_col)

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
        row_lane_idx = [0, 1, 2, 3]
        col_lane_idx = []  # 列车道线索引

        # 局部窗口参数配置 -----------------------------------------------------------------
        local_width_row = 14  # 行方向预测时考虑的局部窗口宽度（左右各扩展14个网格）   1-100范围内选择14-86
        local_width_col = 14  # 列方向预测时考虑的局部窗口宽度
        min_lanepts_row = 3  # 行车道线有效的最小点数阈值
        min_lanepts_col = 3  # 列车道线有效的最小点数阈值

        # 处理行车道线（垂直方向车道线）---------------------------------------------
        for i in row_lane_idx:
            tmp = []
            # 存在性判断：存在车道线
            if valid_row[0, :, i].sum() > 3:
                # 遍历每个行锚点
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:  # 当前锚点存在车道线
                        # 在预测位置周围创建索引窗口（提高位置精度）
                        all_ind = torch.tensor(list(range(
                            max(0, max_indices_row[0, k, i] - local_width_row),
                            min(num_grid_row - 1, max_indices_row[0, k, i] + local_width_row) + 1
                        )))

                        # 计算加权平均位置 -------------------------------------------------
                        # 使用softmax计算窗口内各位置的权重
                        # 公式：out_tmp = Σ(softmax(loc_row[all_ind]) * all_ind) + 0.5
                        out_tmp = (loc_row[0, all_ind, k, i].softmax(0) * all_ind.float()).sum() + 0.5
                        # 将归一化位置映射回原图宽度
                        out_tmp = out_tmp / (num_grid_row - 1) * 800
                        # 计算对应y坐标（行锚点位置）
                        y_coord = int(self.row_anchor[k] * 320)
                        tmp.append((int(out_tmp), y_coord))
                coords.append(tmp)

        # 处理列车道线（水平方向车道线）---------------------------------------------
        for i in col_lane_idx:
            tmp = []
            # 存在性判断：超过1/4的锚点认为存在
            if valid_col[0, :, i].sum() > 3:
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
            img : numpy.ndarray - 输入图像，BGR颜色格式，形状为[H(高度), W(宽度), 3]  任意分辨率的输入
        返回：
            coords : list - 车道线坐标列表，每个元素为[(x1,y1), (x2,y2),...]表示单条车道线
        """
        # ============================ 视频帧预处理 ==============================
        # 阶段目标：将输入帧适配到模型训练时的视野范围和比例

        # 1. 基准尺寸调整（双线性插值平衡效率与质量）
        # OpenCV的resize参数顺序为（宽，高），输出形状变为（高=903，宽=800）
        resized_frame = cv2.resize(img, (800, 903))

        # 2. 关键区域提取（380:700行，去除天空和引擎盖干扰）# 针对摄像头修改
        roi_frame = resized_frame[380:700, :, :]  # 获取高度320px的ROI

        # 3. 保留原始ROI用于可视化（避免后续处理影响显示）
        visual_frame = roi_frame.copy()  # 类型：numpy.ndarray (320, 800, 3)

        # ============================ 模型输入预处理 =============================
        # 阶段目标：生成符合模型输入规范的张量

        # 1. 裁剪顶部区域（根据训练配置截取有效区域）
        # 输入形状（高=320, 宽=800）→ 输出形状（高=320-cut_height, 宽=800）
        model_input = roi_frame[self.cut_height:, :, :]  # 形状变为 (320-cut_height, 800, 3)  ((320-64), 800, 3)

        # 2. 尺寸标准化（双三次插值保持几何精度）
        # 调整到模型输入尺寸（宽=self.input_width, 高=self.input_height）
        model_input = cv2.resize(
            model_input,
            (self.input_width, self.input_height),
            interpolation=cv2.INTER_CUBIC
        )  # 输出形状变为（高，宽，3）即（self.input_height, self.input_width, 3）

        # visual_frame = model_input.copy()

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
        # 将模型输出坐标转换为原始ROI图像的坐标（假设pred2coords已处理缩放）
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
    parser.add_argument('--config_path', default='configs/tusimple_res18.py',
                        help='模型配置文件路径', type=str)
    parser.add_argument('--onnx_path', default='weights/tusimple_res18.onnx',
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
    detector = UFLDv2_ONNX(args.onnx_path, args.config_path, (800, 320))

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
