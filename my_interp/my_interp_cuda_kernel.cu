// 头文件包含
#include <torch/extension.h>  // PyTorch扩展头文件
#include <cuda.h>            // CUDA头文件
#include <cuda_runtime.h>    // CUDA运行时API
#include <vector>            // STL向量容器
#include <cmath>            // 数学函数

// 匿名命名空间，防止符号冲突
namespace {

// CUDA核函数，实现插值计算
__global__ void my_interp_cuda_kernel(
    // 输入参数说明：
    const torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> d_input,  // 输入4D张量 [batch, lanes, points, 2]
    const torch::PackedTensorAccessor<float,1,torch::RestrictPtrTraits,size_t> d_interp_loc, // 插值位置 [new_points]
    torch::PackedTensorAccessor<float,4,torch::RestrictPtrTraits,size_t> d_output, // 输出张量 [batch, lanes, new_points, 2]
    int direction  // 插值方向 (0:水平，1:垂直)
) {
    // ------------------------- 1. 计算线程索引和全局索引 --------------------------
    // 计算block和thread的索引
    int blockId = blockIdx.x + gridDim.x * blockIdx.y; // 二维网格到一维的block索引
    int idx = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    // ------------------------- 2. 张量维度参数 --------------------------
    const int bs = d_input.size(0); // batch size
    const int ls = d_input.size(1); // 车道线数量
    const int cs = d_input.size(2); // 原始点数量
    const int newcs = d_interp_loc.size(0); // 新插值点数量

    // ------------------------- 3. 边界检查 --------------------------
    if (idx >= bs * ls * newcs) return; // 超出有效范围则退出

    // ------------------------- 4. 计算当前线程处理的元素索引 --------------------------
    int newcs_idx = idx % newcs;        // 当前插值点索引
    int ls_idx = (idx / newcs) % ls;    // 车道线索引
    int bs_idx = (idx / newcs) / ls;    // batch索引

    // ------------------------- 5. 获取当前插值位置 --------------------------
    float current_loc = d_interp_loc[newcs_idx];
    d_output[bs_idx][ls_idx][newcs_idx][1-direction] = current_loc; // 保存位置信息

    // ------------------------- 6. 查找包围当前插值位置的区间 --------------------------
    int pos = -1;
    for (int i = cs - 1; i > 0; --i) {
        // 跳过无效点（坐标小于0）
        if (d_input[bs_idx][ls_idx][i][direction] < 0 || 
            d_input[bs_idx][ls_idx][i-1][direction] < 0) 
            continue;

        // 检查当前插值点是否在区间内
        float val_curr = d_input[bs_idx][ls_idx][i][1-direction];
        float val_prev = d_input[bs_idx][ls_idx][i-1][1-direction];
        if ((val_curr - current_loc) * (val_prev - current_loc) <= 0) {
            pos = i;
            break;
        }
    }

    // ------------------------- 7. 线性插值计算 --------------------------
    if (pos != -1) {
        // 计算插值权重
        float len = fabs(d_input[bs_idx][ls_idx][pos][1-direction] - 
                        d_input[bs_idx][ls_idx][pos-1][1-direction]);
        float part1 = fabs(d_input[bs_idx][ls_idx][pos][1-direction] - current_loc);
        float factor1 = 1.0f - part1 / len;
        float factor2 = 1.0f - factor1;

        // 线性插值计算
        float value = d_input[bs_idx][ls_idx][pos][direction] * factor1 +
                     d_input[bs_idx][ls_idx][pos-1][direction] * factor2;

        // 写入结果
        d_output[bs_idx][ls_idx][newcs_idx][direction] = value;
    }
}

} // namespace

// 包装函数：CUDA核函数的调用入口
torch::Tensor my_interp_cuda(
    torch::Tensor input,         // 输入张量 [batch, lanes, points, 2]
    torch::Tensor interp_loc,    // 插值位置 [new_points]
    int direction               // 插值方向 (0:水平，1:垂直)
) {
    // ------------------------- 1. 参数检查 --------------------------
    TORCH_CHECK(input.dim() == 4, "输入必须是4D张量");
    TORCH_CHECK(interp_loc.dim() == 1, "插值位置必须是1D张量");
    
    // ------------------------- 2. 准备输出张量 --------------------------
    auto options = torch::TensorOptions()
        .dtype(torch::kFloat32)
        .device(input.device())
        .requires_grad(false);
    
    const int bs = input.size(0);
    const int ls = input.size(1);
    const int newcs = interp_loc.size(0);
    
    auto output = torch::full({bs, ls, newcs, 2}, -1.0, options);

    // ------------------------- 3. 计算CUDA核参数 --------------------------
    const int threads = 1024; // 每个block的线程数
    const int numElements = bs * ls * newcs;
    const int blocks = (numElements + threads - 1) / threads;

    // ------------------------- 4. 启动CUDA核函数 --------------------------
    my_interp_cuda_kernel<<<blocks, threads>>>(
        input.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
        interp_loc.packed_accessor<float,1,torch::RestrictPtrTraits,size_t>(),
        output.packed_accessor<float,4,torch::RestrictPtrTraits,size_t>(),
        direction
    );

    return output;
}
