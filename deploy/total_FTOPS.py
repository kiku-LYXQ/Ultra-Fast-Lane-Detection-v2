"""
ONNX模型计算工具（支持FLOPs统计与推理测速）
功能：
1. 计算模型理论计算量(FLOPs)
2. 测量CPU/GPU实际推理时间（含最佳/最差耗时）
依赖：pip install onnxruntime onnx onnx-tool
"""

import argparse
import time
import torch
import numpy as np
import onnxruntime as ort
from onnx_tool import model_profile

def measure_inference_time(onnx_path, input_shape, use_gpu=True):
    """测量单次推理耗时，返回平均/最佳/最差耗时"""
    # 初始化推理会话
    providers = ['CUDAExecutionProvider'] if use_gpu else ['CPUExecutionProvider']
    session = ort.InferenceSession(onnx_path, providers=providers)

    # 获取输入名称并生成随机数据
    input_name = session.get_inputs()[0].name
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # 预热运行（消除初始化开销）
    for _ in range(10):
        _ = session.run(None, {input_name: input_data})

    # 正式测速
    times = []
    for _ in range(100):
        start = time.perf_counter()
        _ = session.run(None, {input_name: input_data})
        times.append(time.perf_counter() - start)

    # 转换为毫秒并计算统计值
    times_ms = np.array(times) * 1000
    return {
        'avg': np.mean(times_ms),
        'min': np.min(times_ms),
        'max': np.max(times_ms)
    }

def main():
    parser = argparse.ArgumentParser(description='ONNX模型分析工具')
    parser.add_argument('--model', type=str, required=True, help='ONNX模型路径')
    parser.add_argument('--shape', nargs='+', type=int, default=[1,3,320,1600],
                      help='输入尺寸 (NCHW格式)，例如：1 3 320 1600')
    args = parser.parse_args()

    try:
        # 步骤1：FLOPs分析
        print("\n=== 理论计算量分析 ===")
        model_profile(args.model)

        # 步骤2：CPU测速
        print("\n=== CPU推理测速 ===")
        cpu_stats = measure_inference_time(args.model, args.shape, use_gpu=False)
        print(f"平均耗时: {cpu_stats['avg']:.2f} ms")
        print(f"最佳耗时: {cpu_stats['min']:.2f} ms")
        print(f"最差耗时: {cpu_stats['max']:.2f} ms")

        # 步骤3：GPU测速（如果可用）
        try:
            print("\n=== GPU推理测速 ===")
            gpu_stats = measure_inference_time(args.model, args.shape, use_gpu=True)
            print(f"平均耗时: {gpu_stats['avg']:.2f} ms")
            print(f"最佳耗时: {gpu_stats['min']:.2f} ms")
            print(f"最差耗时: {gpu_stats['max']:.2f} ms")
        except RuntimeError as e:
            print(f"GPU测速失败: {str(e)}（请确认安装onnxruntime-gpu包）")

    except Exception as e:
        print(f"运行错误: {str(e)}")

if __name__ == "__main__":
    main()