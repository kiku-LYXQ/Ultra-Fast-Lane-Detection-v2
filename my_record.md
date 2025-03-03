```markdown
# Ultra-Fast-Lane-Detection-v2 ONNX 推理指南

🚗 基于ONNX Runtime的车道线检测实现 | [GitHub项目](https://github.com/your_project_url)

---

## 环境配置

### 步骤1：安装基础依赖
```bash
# 安装NVIDIA驱动（推荐470+版本）
sudo apt-get install nvidia-driver-470

# 安装CUDA 11.3（需与PyTorch版本匹配）
wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda_11.3.0_465.19.01_linux.run
sudo sh cuda_11.3.0_465.19.01_linux.run

# 安装PyTorch 1.12.1 + CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

### 步骤2：安装项目依赖
```bash
# ONNX Runtime GPU版（需与CUDA版本匹配）
pip install onnxruntime-gpu==1.16.0

# 其他依赖
pip install opencv-python numpy==1.21.0 tqdm
```

---

## 项目配置

### 文件结构准备
```bash
export PROJECT_ROOT=/path/to/your/project  # 设置项目根目录
mkdir -p $PROJECT_ROOT/{configs,weights,utils}
```

### 关键文件准备
1. 下载模型文件至 `weights/` 目录：
   ```bash
   wget https://example.com/culane_res34.onnx -P $PROJECT_ROOT/weights/
   ```
2. 配置文件放置到 `configs/` 目录

---

## 环境变量配置
```bash
# 解决libstdc++兼容性问题
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6

# 添加项目根目录到Python路径
export PYTHONPATH=$PYTHONPATH:$PROJECT_ROOT
```

---

## 执行推理

### 基础命令
```bash
python $PROJECT_ROOT/deploy/CulaneRes34Onnx_infer.py \
  --config_path  $PROJECT_ROOT/configs/culane_res34.py \
  --onnx_path $PROJECT_ROOT/weights/culane_res34.onnx \
  --video_path $PROJECT_ROOT/example.mp4
```

### 参数说明表
| 参数 | 类型 | 必选 | 默认值 | 说明 |
|------|------|------|--------|-----|
| `--config_path` | str  | ✓ | 无 | 模型配置文件路径 |
| `--onnx_path`   | str  | ✓ | 无 | ONNX模型文件路径 |
| `--video_path`  | str  | ✓ | 无 | 输入视频路径 |
| `--ori_size`    | tuple| ✕ | (1600,320) | 原始图像尺寸 (宽,高) |

---

## 常见问题处理

### 1. CUDA内存不足
```bash
# 方案1：降低输入分辨率
--ori_size 1280 720

# 方案2：使用FP16量化模型
wget https://example.com/culane_res34_fp16.onnx
```

### 2. libstdc++版本冲突
```bash
# 永久解决方案（写入.bashrc）
echo 'export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6' >> ~/.bashrc
source ~/.bashrc
```

---

## 可视化效果

输入 / 输出对比示例：
```
原始帧 → 检测结果
[![示例图片](http://example.com/demo.jpg)](http://example.com/demo_video.mp4)
```

---

> 📌 **注意事项**  
> 1. 确保视频的ROI区域与代码中`img[380:700, :, :]`匹配  
> 2. 首次运行前执行 `chmod +x deploy/*.py` 赋予执行权限  
> 3. 推荐使用1280x720以上分辨率视频以获得最佳效果
```