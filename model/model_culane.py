import torch
from model.backbone import resnet
from utils.common import initialize_weights
from model.seg_model import SegHead


class parsingNet(torch.nn.Module):
    def __init__(self, pretrained=True, backbone='50', num_grid_row=None,
                 num_cls_row=None, num_grid_col=None, num_cls_col=None,
                 num_lane_on_row=None, num_lane_on_col=None, use_aux=False,
                 input_height=None, input_width=None, fc_norm=False):
        """
        维度变化示意图：
        输入图像 → ResNet特征提取 → 特征降维 → 展平 → 全连接层 → 输出拆分
        [B,3,H,W] → [B,2048,H/32,W/32] → [B,8,H/32,W/32] → [B,D] → [B,T] → 多任务输出
        """
        super(parsingNet, self).__init__()

        # 参数初始化
        self.num_grid_row = num_grid_row  # 行方向网格数 (e.g. 72)
        self.num_cls_row = num_cls_row  # 行方向分类数 (e.g. 56)
        self.num_grid_col = num_grid_col  # 列方向网格数 (e.g. 81)
        self.num_cls_col = num_cls_col  # 列方向分类数 (e.g. 41)
        self.num_lane_on_row = num_lane_on_row  # 行车道线数 (e.g. 4)
        self.num_lane_on_col = num_lane_on_col  # 列车道线数 (e.g. 4)
        self.use_aux = use_aux  # 是否使用辅助分割头

        # 输出维度计算 ----------------------------------------------------------
        # 维度计算示例（假设参数为论文默认值）：
        # loc_row: 72x56x4=16128, loc_col: 81x41x4=13284
        # exist_row: 2x56x4=448, exist_col: 2x41x4=328
        # total_dim = 16128+13284+448+328=30188
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row  # 行位置预测维度
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col  # 列位置预测维度
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row  # 行存在性预测维度（二分类）
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col  # 列存在性预测维度（二分类）
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4  # 总输出维度

        # 输入维度计算（ResNet下采样32倍）-----------------------------------------
        # 输入示例：288x512 → 288//32=9, 512//32=16 → 9x16=144
        # 1x1卷积将通道数降为8 → 144x8=1152
        self.input_dim = (input_height // 32) * (input_width // 32) * 8

        # 主干网络 ------------------------------------------------------------
        # ResNet-50输出维度：[B, 2048, H/32, W/32]
        # ResNet-34输出维度：[B, 512, H/32, W/32]
        self.model = resnet(backbone, pretrained=pretrained)

        # 分类头 -------------------------------------------------------------
        mlp_mid_dim = 2048  # MLP隐藏层维度
        self.cls = torch.nn.Sequential(
            # LayerNorm层（可选）: [B,1152] → [B,1152] (参数：1152*2=2304)
            torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),

            # 全连接层1: 1152 → 2048 (参数：1152*2048 + 2048 = 2,360,320)
            torch.nn.Linear(self.input_dim, mlp_mid_dim),

            # ReLU激活: 保持维度不变 [B,2048] → [B,2048]
            torch.nn.ReLU(),

            # 全连接层2: 2048 → total_dim (参数：2048*30188 + 30188 ≈ 61.8M)
            torch.nn.Linear(mlp_mid_dim, self.total_dim)
        )

        # 特征降维卷积 --------------------------------------------------------
        # 1x1卷积用于通道降维（无空间下采样）
        # ResNet-50: [B,2048,H/32,W/32] → [B,8,H/32,W/32] (参数：2048*8=16,384)
        # ResNet-34: [B,512,H/32,W/32] → [B,8,H/32,W/32] (参数：512*8=4,096)
        self.pool = torch.nn.Conv2d(
            512 if backbone in ['34', '18', '34fca'] else 2048,
            8,
            kernel_size=1
        )

        # 辅助分割头 ---------------------------------------------------------
        if self.use_aux:
            # 输出通道数：行车道线数 + 列车道线数 (e.g. 4+4=8)
            # 输出维度：[B, num_lanes*2, H, W]
            self.seg_head = SegHead(backbone, num_lane_on_row + num_lane_on_col)

        initialize_weights(self.cls)

    def forward(self, x):
        """
        前向传播维度变化示例（假设输入为[2,3,288,512]）：
        x2: [2,256,36,64]  (ResNet layer2输出)
        x3: [2,512,18,32]  (ResNet layer3输出)
        fea: [2,2048,9,16] (ResNet最终输出)
        """
        # 主干网络前传
        x2, x3, fea = self.model(x)  # fea维度：[B, C, H/32, W/32]

        # 辅助分割头
        if self.use_aux:
            seg_out = self.seg_head(x2, x3, fea)  # 输出维度：[B, num_lanes*2, H, W]

        # 特征处理流程 --------------------------------------------------------
        # 降维卷积：2048/512 → 8通道
        fea = self.pool(fea)  # 维度：[B,8,9,16]

        # 展平操作：将空间维度展平为向量
        fea = fea.view(-1, self.input_dim)  # 维度：[2, 9*16*8=1152]

        # 分类头处理
        out = self.cls(fea)  # 维度变化：1152 → 2048 → 30188 ([2,30188])

        # 输出拆分 -----------------------------------------------------------
        pred_dict = {
            # loc_row: [2,72,56,4] (72网格 * 56类别 * 4车道线)
            'loc_row': out[:, :self.dim1].view(-1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row),

            # loc_col: [2,81,41,4] (81网格 * 41类别 * 4车道线)
            'loc_col': out[:, self.dim1:self.dim1 + self.dim2].view(-1, self.num_grid_col, self.num_cls_col,
                                                                    self.num_lane_on_col),

            # exist_row: [2,2,56,4] (每个位置二分类概率)
            'exist_row': out[:, self.dim1 + self.dim2:self.dim1 + self.dim2 + self.dim3].view(-1, 2, self.num_cls_row,
                                                                                              self.num_lane_on_row),

            # exist_col: [2,2,41,4]
            'exist_col': out[:, -self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)
        }

        if self.use_aux:
            pred_dict['seg_out'] = seg_out  # 分割输出：[2,8,288,512]

        return pred_dict


def get_model(cfg):
    """模型构建入口函数
    返回配置好的模型实例（自动转移到GPU）
    典型配置参数示例：
        cfg.backbone = '50'          # ResNet-50
        cfg.num_cell_row = 72        # 行网格数
        cfg.num_row = 56             # 行分类数
        cfg.num_cell_col = 81        # 列网格数
        cfg.num_col = 41             # 列分类数
        cfg.num_lanes = 4            # 车道线数量
        cfg.use_aux = True           # 启用辅助分割
        cfg.train_height = 288       # 输入高度
        cfg.train_width = 512        # 输入宽度
    """
    return parsingNet(
        pretrained=True,
        backbone=cfg.backbone,
        num_grid_row=cfg.num_cell_row,
        num_cls_row=cfg.num_row,
        num_grid_col=cfg.num_cell_col,
        num_cls_col=cfg.num_col,
        num_lane_on_row=cfg.num_lanes,
        num_lane_on_col=cfg.num_lanes,
        use_aux=cfg.use_aux,
        input_height=cfg.train_height,
        input_width=cfg.train_width,
        fc_norm=cfg.fc_norm
    ).cuda()
