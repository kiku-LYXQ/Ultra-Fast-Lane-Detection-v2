"""
基于ONNX Runtime的Ultra-Fast-Lane-Detection-v2推理实现（无PyTorch依赖版）
树莓派优化要点：
1. 完全使用NumPy替代PyTorch操作
2. 限制ONNX Runtime线程数降低CPU占用
3. 使用OpenCV原生接口提升图像处理效率
"""

import cv2
import numpy as np
import onnxruntime as ort
import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
from utils.config import Config

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
    def __init__(self, onnx_path, config_path, ori_size):
        # 配置ONNX Runtime参数 -------------------------------------------------
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4  # 限制计算线程数
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            onnx_path,
            providers=['CPUExecutionProvider'],  # 强制使用CPU
            sess_options=options
        )

        # 输入输出配置 --------------------------------------------------------
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]

        # 加载配置文件参数 ----------------------------------------------------
        cfg = Config.fromfile(config_path)
        self.ori_img_w, self.ori_img_h = ori_size
        self.cut_height = int(cfg.train_height * (1 - cfg.crop_ratio))
        self.input_width = cfg.train_width
        self.input_height = cfg.train_height
        self.num_row = cfg.num_row
        self.num_col = cfg.num_col

        # 预计算锚点坐标 ------------------------------------------------------
        self.row_anchor = np.linspace(160, 710, cfg.num_row) / 720
        self.col_anchor = np.linspace(0, 1, cfg.num_col)

    def numpy_softmax(self, x, axis=-1):
        """数值稳定的Softmax实现"""
        e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return e_x / e_x.sum(axis=axis, keepdims=True)

    def pred2coords(self, pred):
        """纯NumPy实现的后处理逻辑"""
        loc_row, loc_col, exist_row, exist_col = pred  # 解包模型输出

        # 最大概率索引获取（numpy实现）----------------------------------------
        # loc_row形状说明：[batch, num_grid_row, num_cls_row, num_lane]
        max_indices_row = np.argmax(loc_row, axis=1)  # 沿num_grid_row维度取最大值
        valid_row = np.argmax(exist_row, axis=1)  # 存在性判断（0/1）

        max_indices_col = np.argmax(loc_col, axis=1)
        valid_col = np.argmax(exist_col, axis=1)

        coords = []  # 存储所有车道线的坐标
        row_lane_idx = [0, 1, 2, 3]
        col_lane_idx = []  # 列车道线索引

        # 局部窗口参数配置 -----------------------------------------------------------------
        local_width_row = 14  # 行方向预测时考虑的局部窗口宽度（左右各扩展14个网格）   1-100范围内选择14-86
        local_width_col = 14  # 列方向预测时考虑的局部窗口宽度
        min_lanepts_row = 3  # 行车道线有效的最小点数阈值
        min_lanepts_col = 3  # 列车道线有效的最小点数阈值

        # 行车道线处理（垂直方向）---------------------------------------------
        for i in row_lane_idx:
            tmp = []
            # 存在性判断：有效锚点超过半数则视为存在车道线
            if valid_row[0, :, i].sum() > self.num_row / 2:
                # 遍历每个行锚点（垂直方向采样点）
                for k in range(valid_row.shape[1]):
                    if valid_row[0, k, i]:  # 当前锚点存在车道线
                        # 生成索引窗口（在预测位置周围扩展）
                        current_max = max_indices_row[0, k, i]
                        start = max(0, current_max - self.input_width)
                        end = min(loc_row.shape[1] - 1, current_max + self.input_width) + 1
                        all_ind = np.arange(start, end)

                        # 加权平均计算（关键精度提升步骤）-----------------------
                        # 公式：out_tmp = Σ(softmax(score) * index) + 0.5
                        weights = softmax(loc_row[0, all_ind, k, i], axis=-1)
                        out_tmp = np.sum(weights * all_ind) + 0.5  # +0.5用于四舍五入

                        # 坐标映射：将归一化位置转换为实际像素坐标
                        # 映射公式：实际坐标 = 预测位置 / (网格数-1) × 原图宽度
                        out_tmp = out_tmp / (loc_row.shape[1] - 1) * self.ori_img_w

                        # 计算对应y坐标（基于行锚点位置）
                        y_coord = int(self.row_anchor[k] * self.ori_img_h)
                        tmp.append((int(out_tmp), y_coord))
                coords.append(tmp)

                # 列车道线处理（水平方向，逻辑类似）-------------------------------------
                for i in col_lane_idx:
                    tmp = []
                    # 存在性阈值稍低（1/4有效锚点即视为存在）
                    if valid_col[0, :, i].sum() > self.num_col / 4:
                        for k in range(valid_col.shape[1]):
                            if valid_col[0, k, i]:
                                current_max = max_indices_col[0, k, i]
                                start = max(0, current_max - self.input_width)
                                end = min(loc_col.shape[1] - 1, current_max + self.input_width) + 1
                                all_ind = np.arange(start, end)

                                weights = softmax(loc_col[0, all_ind, k, i], axis=-1)
                                out_tmp = np.sum(weights * all_ind) + 0.5

                                # 注意此处映射到高度方向
                                out_tmp = out_tmp / (loc_col.shape[1] - 1) * self.ori_img_h

                                x_coord = int(self.col_anchor[k] * self.ori_img_w)
                                tmp.append((x_coord, int(out_tmp)))
                        coords.append(tmp)

        return coords

    def forward(self, img):
        """优化后的推理流程"""
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

        # 推理
        outputs = self.session.run(self.output_names, {self.input_name: input_tensor})

        # 后处理
        coords = self.pred2coords(outputs)

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
                cv2.circle(visual_frame, (x, y), radius=1, color=(0, 255, 0), thickness=-1)

        # 实时显示检测结果
        cv2.imshow("Lane Detection", visual_frame)
        cv2.waitKey(1)  # 允许图像窗口更新

        return coords


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='configs/tusimple_res18.py')
    parser.add_argument('--onnx_path', default='weights/tusimple_res18.onnx')
    parser.add_argument('--video_path', default='example.mp4')
    parser.add_argument('--ori_size', type=int, nargs=2, default=[800, 320])
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    detector = UFLDv2_ONNX(args.onnx_path, args.config_path, args.ori_size)

    cap = cv2.VideoCapture(args.video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        detector.forward(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()