"""
CULane数据集标注缓存生成器
功能：将车道线标注转换为快速读取的JSON格式，加速训练数据加载
执行命令示例：python scripts/cache_culane_ponits.py --root /path/to/culane
"""

import os
import cv2
import numpy as np
import tqdm  # 进度条显示
import json
import argparse


def get_args():
    """命令行参数解析器"""
    parser = argparse.ArgumentParser(description='CULane标注缓存生成工具')
    parser.add_argument('--root', required=True,
                        help='CULane数据集根目录路径，应包含list目录和图片数据')
    return parser


if __name__ == '__main__':
    # 参数解析
    args = get_args().parse_args()
    culane_root = args.root  # 数据集根目录

    # 读取训练集标注列表
    train_list = os.path.join(culane_root, 'list/train_gt.txt')
    with open(train_list, 'r') as fp:
        res = fp.readlines()  # 读取所有标注行

    cache_dict = {}  # 存储最终缓存数据的字典

    # 进度条遍历每个标注项
    for line in tqdm.tqdm(res):
        info = line.split(' ')  # 分割每行数据

        # 标签路径处理 ---------------------------------------------------------
        label_path = os.path.join(culane_root, info[1][1:])  # 去除路径开头多余斜杠
        label_img = cv2.imread(label_path)[:, :, 0]  # 读取标签图像（单通道）

        # 车道线坐标文件路径处理 ------------------------------------------------
        txt_path = info[0][1:].replace('jpg', 'lines.txt')  # 原始图片路径转标注路径
        txt_path = os.path.join(culane_root, txt_path)
        lanes = open(txt_path, 'r').readlines()  # 读取车道线坐标数据

        # 初始化数据存储结构 ---------------------------------------------------
        """
        all_points 结构说明：
        - 第1维度：4条车道线（CULane最大车道数）
        - 第2维度：35个预定义行锚点（从250到590，间隔10像素）
        - 第3维度：2个坐标值（x,y）
        初始化x值为-99999表示无效点
        """
        all_points = np.zeros((4, 35, 2), dtype=np.float64)
        # 预定义纵向坐标（y轴位置）
        the_anno_row_anchor = np.array([
            250, 260, 270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400,
            410, 420, 430, 440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560, 570, 580, 590
        ])
        # 填充y坐标
        all_points[:, :, 1] = np.tile(the_anno_row_anchor, (4, 1))  # 复制到所有车道线
        all_points[:, :, 0] = -99999  # 初始化x坐标为无效值

        # 处理每条车道线 ------------------------------------------------------
        for lane_idx, lane in enumerate(lanes):
            ll = lane.strip().split(' ')  # 分割坐标点字符串
            point_x = ll[::2]  # 提取x坐标（偶数索引）
            point_y = ll[1::2]  # 提取y坐标（奇数索引）

            # 计算车道线中点用于确定车道顺序 --------------------------------------
            mid_x = int(float(point_x[int(len(point_x) / 2)]))  # 取中间点x坐标
            mid_y = int(float(point_y[int(len(point_x) / 2)]))  # 取中间点y坐标
            lane_order = label_img[mid_y - 1, mid_x - 1]  # 从标签图像获取车道编号（1-4）

            # 调试断点（当遇到未知车道编号时暂停）
            if lane_order == 0:
                import pdb;

                pdb.set_trace()

            # 填充有效坐标点 ---------------------------------------------------
            for i in range(len(point_x)):
                p1x = float(point_x[i])  # 当前点x坐标
                # 计算对应的行锚点索引：(实际y坐标 - 基准250) / 10间距
                pos = (int(point_y[i]) - 250) // 10
                # 限制索引范围在0-34之间
                pos = max(0, min(34, pos))
                # 存储有效x坐标
                all_points[lane_order - 1, int(pos), 0] = p1x

        # 保存处理结果 --------------------------------------------------------
        cache_dict[info[0][1:]] = all_points.tolist()  # 使用图片相对路径作为键

    # 输出缓存文件 -----------------------------------------------------------
    output_path = os.path.join(culane_root, 'culane_anno_cache.json')
    with open(output_path, 'w') as f:
        json.dump(cache_dict, f)  # 写入JSON文件
