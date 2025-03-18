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

---

### 核心参数详解

#### 1. **`name`**: 包名称
• **作用**：定义包的名称，安装后可通过`import my_interp`导入。
• **注意**：需与`CUDAExtension`中的模块名称一致。

#### 2. **`ext_modules`**: 扩展模块列表
• **`CUDAExtension`参数**：
  • **第一个参数**：生成的模块名称（如`"my_interp"`）。
  • **`sources`**: 源文件列表（C++/CUDA代码文件）。
  • **`extra_compile_args`**: 自定义编译选项（可选）。

#### 3. **`cmdclass`**: 构建扩展命令
• **`BuildExtension`的作用**：
  • 自动处理CUDA路径、PyTorch头文件、库链接。
  • 支持混合编译C++和CUDA代码。

---

### 实际构建流程

#### 步骤1：编写源代码
• **C++包装代码** (`my_interp_cuda.cpp`):  
  定义Python绑定（如`torch::Tensor`接口），调用CUDA函数。
• **CUDA核函数** (`my_interp_cuda_kernel.cu`):  
  实现GPU加速的核心计算逻辑。

#### 步骤2：运行构建命令
在终端执行以下命令：
```bash
# 安装到当前Python环境
python setup.py install

# 或开发模式（代码修改后无需重新安装）
python setup.py develop
```

#### 步骤3：验证安装
在Python中测试导入：
```python
import my_interp  # 导入编译后的模块
output = my_interp.my_cuda_function(input_tensor)  # 调用CUDA扩展
```

---

### 关键机制说明

#### 1. **混合编译C++和CUDA**
• **`.cpp`文件**：由C++编译器（如`g++`）处理。
• **`.cu`文件**：由CUDA编译器（`nvcc`）处理。
• **PyTorch的`BuildExtension`**：自动生成编译命令，确保正确链接PyTorch库。

#### 2. **编译优化选项**
• **`-O3`**: 最高级别优化（权衡编译速度与性能）。
• **其他常用选项**：
  ```python
  extra_compile_args={
      "cxx": ["-fopenmp"],  # 启用OpenMP并行
      "nvcc": ["-arch=sm_70"]  # 指定GPU计算能力（如Volta架构）
  }
  ```

#### 3. **多扩展模块支持**
若需编译多个独立扩展：
```python
ext_modules=[
    CUDAExtension("module1", sources=["mod1.cpp", "mod1_kernel.cu"]),
    CUDAExtension("module2", sources=["mod2.cpp", "mod2_kernel.cu"])
]
```

---

### 典型应用场景

| 场景                  | 描述                              |
|-----------------------|----------------------------------|
| **自定义CUDA算子**    | 实现PyTorch不支持的GPU加速操作（如特殊插值） |
| **性能关键代码优化**  | 替代Python循环，提升计算密集型任务速度     |
| **C++库集成**         | 将现有C++/CUDA库封装为Python可调用接口    |

---

通过以上配置，`setup`函数将自动完成从C++/CUDA代码到Python模块的编译和安装，极大简化了混合语言开发的复杂性。

好的，我现在需要帮助用户理解这个Python函数`get_model(cfg)`的使用方法，特别是用户提到的模块`importlib.import_module('model.model_'+cfg.dataset.lower()).get_model(cfg)`的具体用法。用户希望我通过举例说明这个模块是如何工作的，并且可能想知道如何在实际项目中应用它。

首先，我需要理解这个函数的结构。这个函数接受一个配置对象`cfg`，然后动态导入一个模块，该模块的路径由`cfg.dataset`的值决定。例如，如果`cfg.dataset`是"MNIST"，那么导入的模块就是`model.model_mnist`，然后调用该模块中的`get_model`函数，并将`cfg`传递给它，最终返回得到的模型。

用户可能需要知道如何配置`cfg`对象，以及如何组织他们的模型模块。接下来，我应该考虑如何通过一个实际的例子来展示这个函数的使用流程，包括目录结构、配置文件、模型模块的实现以及调用过程。

我需要先假设一个项目结构，比如：

• 项目根目录/
  • config.py（定义配置类）
  • model/
    ◦ __init__.py（空文件）
    ◦ model_mnist.py（MNIST模型定义）
    ◦ model_cifar10.py（CIFAR-10模型定义）
  • main.py（主程序，使用get_model函数）

然后，我需要编写每个文件的内容。例如，`config.py`中可能有一个配置类，包含`dataset`属性。`model_mnist.py`和`model_cifar10.py`中各自定义了一个`get_model`函数，根据配置返回对应的模型。

接下来，我需要说明在`main.py`中如何导入配置和`get_model`函数，并调用它来获取模型。同时，用户可能需要了解如何动态切换不同的数据集模型，只需要修改配置中的`dataset`属性即可。

另外，用户可能会遇到模块导入错误，比如模块名不正确或路径问题，这时候需要提醒他们检查模块的命名和路径是否正确，以及是否包含`__init__.py`文件来使目录成为包。

最后，总结这个设计模式的优点，比如模块化、易于扩展和维护，适用于需要支持多种模型和数据集的场景。




以下是一个完整的示例，说明如何通过 `get_model(cfg)` 函数动态加载不同数据集对应的模型模块：

---

### 1. **项目目录结构**
```bash
your_project/
├── config.py        # 配置文件
├── model/           # 模型模块目录
│   ├── __init__.py
│   ├── model_mnist.py    # MNIST数据集模型
│   └── model_cifar10.py  # CIFAR-10数据集模型
└── main.py          # 主程序
```

---

### 2. **配置文件 `config.py`**
```python
class Config:
    def __init__(self, dataset):
        self.dataset = dataset  # 数据集名称，如"MNIST"或"CIFAR10"
        # 其他配置参数（示例）
        self.input_size = (28, 28) if dataset == "MNIST" else (32, 32, 3)
        self.num_classes = 10

# 示例：创建一个MNIST配置对象
cfg = Config(dataset="MNIST")
```

---

### 3. **模型模块实现**

#### (1) `model/model_mnist.py` (MNIST模型)
```python
import torch.nn as nn

def get_model(cfg):
    class MNISTModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(cfg.input_size[0]*cfg.input_size[1], 512),
                nn.ReLU(),
                nn.Linear(512, cfg.num_classes)
            )
        
        def forward(self, x):
            return self.net(x.view(x.size(0), -1))
    
    return MNISTModel()  # 返回实例化的模型
```

#### (2) `model/model_cifar10.py` (CIFAR-10模型)
```python
import torch.nn as nn

def get_model(cfg):
    class CIFAR10Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(3, 16, 3),  # 输入通道数根据cfg.input_size自动适应
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(16*15*15, cfg.num_classes)
            )
        
        def forward(self, x):
            return self.net(x)
    
    return CIFAR10Model()
```

---

### 4. **主程序 `main.py`**
```python
import importlib
from config import cfg  # 导入配置对象

def get_model(cfg):
    # 动态拼接模块路径，例如：model.model_mnist
    module_name = f"model.model_{cfg.dataset.lower()}"
    # 动态导入模块
    module = importlib.import_module(module_name)
    # 调用模块中的get_model函数并返回模型
    return module.get_model(cfg)

if __name__ == "__main__":
    # 测试MNIST模型
    cfg_mnist = Config(dataset="MNIST")
    model_mnist = get_model(cfg_mnist)
    print("MNIST模型结构:", model_mnist)

    # 测试CIFAR10模型
    cfg_cifar = Config(dataset="CIFAR10")
    model_cifar = get_model(cfg_cifar)
    print("CIFAR10模型结构:", model_cifar)
```

---

### 5. **运行输出示例**
```text
MNIST模型结构: MNISTModel(
  (net): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=10, bias=True)
  )
)

CIFAR10模型结构: CIFAR10Model(
  (net): Sequential(
    (0): Conv2d(3, 16, kernel_size=(3, 3), stride=(1, 1))
    (1): ReLU()
    (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (3): Flatten(start_dim=1, end_dim=-1)
    (4): Linear(in_features=3600, out_features=10, bias=True)
  )
)
```

---

### 关键机制解释
1. **动态模块加载**  
   • 根据`cfg.dataset`的值（如"MNIST"）拼接出模块路径`model.model_mnist`
   • `importlib.import_module()` 动态导入对应模块

2. **统一接口设计**  
   • 所有模型模块必须实现`get_model(cfg)`函数
   • 主程序无需预先知道具体模型类名，通过配置即可切换模型

3. **配置驱动开发**  
   • 通过`cfg`对象传递参数（如输入尺寸、类别数）
   • 模型结构可以根据配置自动调整

---

### 常见问题解决
1. **模块不存在错误**  
   ```python
   ModuleNotFoundError: No module named 'model.model_mnist'
   ```
   • **检查点**：
     ◦ 确认`model`目录存在且包含`__init__.py`
     ◦ 确认模块文件名正确（如`model_mnist.py`）

2. **函数未定义错误**  
   ```python
   AttributeError: module 'model.model_mnist' has no attribute 'get_model'
   ```
   • **检查点**：
     ◦ 确认模型模块中实现了`get_model(cfg)`函数
     ◦ 函数名称拼写是否一致

---

这种设计模式广泛应用于深度学习框架中，支持快速扩展新模型（如MMDetection、HuggingFace Transformers）。当需要新增数据集支持时，只需在`model`目录下添加新的模块文件（如`model_imagenet.py`），无需修改主程序代码。