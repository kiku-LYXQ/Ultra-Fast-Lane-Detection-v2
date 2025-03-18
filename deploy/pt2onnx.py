"""
PyTorch模型转ONNX格式工具
功能：将训练好的车道线检测模型转换为ONNX格式，支持FP32/FP16精度
执行命令示例：
python export_onnx.py --config_path configs/culane_res34.py --model_path weights/culane_res34.pth
"""

import torch
import argparse
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../"))  # 添加父目录到系统路径，确保模块导入正确
import onnxmltools  # ONNX转换工具
from onnxmltools.utils.float16_converter import convert_float_to_float16  # FP16转换器
from utils.common import get_model  # 模型构建函数
from utils.config import Config  # 配置加载类
from utils.dist_utils import dist_print  # 分布式打印工具

def get_args():
    """命令行参数解析器"""
    parser = argparse.ArgumentParser()
    # 配置文件路径（默认使用CULane的ResNet34配置）
    parser.add_argument('--config_path', default='configs/tusimple_res18.py',
                      help='模型配置文件路径', type=str)
    # 预训练模型路径
    parser.add_argument('--model_path', default='weights/tusimple_res18.pth',
                      help='PyTorch模型权重文件路径', type=str)
    # 输出精度选择（FP32或FP16）
    parser.add_argument('--accuracy', default='fp32', choices=['fp16', 'fp32'],
                      help='输出模型的精度类型', type=str)
    # 输入图像尺寸（宽，高）需与训练时一致
    parser.add_argument('--size', default=(800, 320),
                      help='原始输入图像尺寸（宽, 高）', type=tuple)
    return parser.parse_args()

def convert(model, args):
    """模型转换主函数
    参数：
        model : 加载权重的PyTorch模型
        args  : 命令行参数对象
    """
    dist_print('开始模型转换...')

    # 创建虚拟输入数据（batch_size=1, channels=3, height, width）
    images = torch.ones((1, 3, args.size[1], args.size[0])).cuda()

    # 生成ONNX文件路径（将.pth替换为.onnx）
    onnx_path = args.model_path[:-4] + ".onnx"

    with torch.no_grad():
        # Step 1: 导出FP32精度ONNX模型
        torch.onnx.export(
            model,                # 待转换模型
            images,               # 虚拟输入数据
            onnx_path,            # 输出路径
            verbose=False,        # 不显示详细导出信息
            input_names=['input'], # 输入节点名称
            output_names=["loc_row", "loc_col", "exist_row", "exist_col"]  # 输出节点名称
        )
        dist_print("ONNX模型导出成功，保存路径：", onnx_path)

        # Step 2: 如需FP16精度，进行类型转换
        if args.accuracy == 'fp16':
            # 加载导出的FP32模型
            onnx_model = onnxmltools.utils.load_model(onnx_path)
            # 将浮点类型转换为FP16
            onnx_model = convert_float_to_float16(onnx_model)

            # 保存FP16模型
            onnx_half_path = args.model_path[:-4] + "_fp16.onnx"
            onnxmltools.utils.save_model(onnx_model, onnx_half_path)
            dist_print("FP16精度模型保存路径：", onnx_half_path)

if __name__ == "__main__":
    # 启用CuDNN加速
    torch.backends.cudnn.benchmark = True

    # 获取命令行参数
    args = get_args()

    # 加载配置文件 ----------------------------------------------------------------
    cfg = Config.fromfile(args.config_path)
    cfg.batch_size = 1  # 推理时batch_size固定为1

    # 验证模型骨干网络类型（确保使用支持的ResNet变体）
    assert cfg.backbone in ['18', '34', '50', '101', '152',
                          '50next', '101next', '50wide', '101wide']

    # 根据数据集类型设置分类数 -----------------------------------------------------
    if cfg.dataset == 'CULane':
        cls_num_per_lane = 18   # CULane每条车道线预测18个位置点
    elif cfg.dataset == 'Tusimple':
        cls_num_per_lane = 56   # Tusimple数据集为56个
    else:
        raise NotImplementedError

    # 初始化模型 ----------------------------------------------------------------
    net = get_model(cfg)  # 根据配置创建模型实例

    # 加载预训练权重 ------------------------------------------------------------
    state_dict = torch.load(args.model_path, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        # 去除多GPU训练时添加的'module.'前缀（单GPU模型兼容性处理）
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v
    # 加载权重（strict=False允许部分权重不匹配）
    net.load_state_dict(compatible_state_dict, strict=False)

    # 执行转换流程 --------------------------------------------------------------
    convert(net, args)
