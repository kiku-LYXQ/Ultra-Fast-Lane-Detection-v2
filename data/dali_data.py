import torch
import numpy as np
import random
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import nvidia.dali.fn as fn
import json
import os
from nvidia.dali.plugin.pytorch import DALIGenericIterator
from nvidia.dali.plugin.pytorch import LastBatchPolicy
import my_interp
class LaneExternalIterator(object):
    def __init__(self, path, list_path, batch_size=None, shard_id=None, num_shards=None, mode = 'train', dataset_name=None):
        assert mode in ['train', 'test']
        self.mode = mode
        self.path = path
        self.list_path = list_path
        self.batch_size = batch_size
        self.shard_id = shard_id
        self.num_shards = num_shards

        if isinstance(list_path, str):
            with open(list_path, 'r') as f:
                total_list = f.readlines()
        elif isinstance(list_path, list) or isinstance(list_path, tuple):
            total_list = []
            for lst_path in list_path:
                with open(lst_path, 'r') as f:
                    total_list.extend(f.readlines())
        else:
            raise NotImplementedError
        if self.mode == 'train':
            if dataset_name == 'CULane':
                cache_path = os.path.join(path, 'culane_anno_cache.json')
            elif dataset_name == 'Tusimple':
                cache_path = os.path.join(path, 'tusimple_anno_cache.json')
            elif dataset_name == 'CurveLanes':
                cache_path = os.path.join(path, 'train', 'curvelanes_anno_cache.json')
            else:
                raise NotImplementedError

            if shard_id == 0:
                print('loading cached data')
            cache_fp = open(cache_path, 'r')
            self.cached_points = json.load(cache_fp)
            if shard_id == 0:
                print('cached data loaded')

        self.total_len = len(total_list)
    
        self.list = total_list[self.total_len * shard_id // num_shards:
                                self.total_len * (shard_id + 1) // num_shards]
        self.n = len(self.list)

    def __iter__(self):
        self.i = 0
        if self.mode == 'train':
            random.shuffle(self.list)
        return self

    def _prepare_train_batch(self):
        images = []
        seg_images = []
        labels = []

        for _ in range(self.batch_size):
            l = self.list[self.i % self.n]
            l_info = l.split()
            img_name = l_info[0]
            seg_name = l_info[1]

            if img_name[0] == '/':
                img_name = img_name[1:]
            if seg_name[0] == '/':
                seg_name = seg_name[1:]
                
            img_name = img_name.strip()
            seg_name = seg_name.strip()
            
            img_path = os.path.join(self.path, img_name)
            with open(img_path, 'rb') as f:
                images.append(np.frombuffer(f.read(), dtype=np.uint8))

            img_path = os.path.join(self.path, seg_name)
            with open(img_path, 'rb') as f:
                seg_images.append(np.frombuffer(f.read(), dtype=np.uint8))

            points = np.array(self.cached_points[img_name])
            labels.append(points.astype(np.float32))

            self.i = self.i + 1
            
        return (images, seg_images, labels)

    
    def _prepare_test_batch(self):
        images = []
        names = []
        for _ in range(self.batch_size):
            img_name = self.list[self.i % self.n].split()[0]

            if img_name[0] == '/':
                img_name = img_name[1:]
            img_name = img_name.strip()

            img_path = os.path.join(self.path, img_name)

            with open(img_path, 'rb') as f:
                images.append(np.frombuffer(f.read(), dtype=np.uint8))
            names.append(np.array(list(map(ord,img_name))))
            self.i = self.i + 1
            
        return images, names

    def __next__(self):
        if self.i >= self.n:
            self.__iter__()
            raise StopIteration
        if self.mode == 'train':
            res = self._prepare_train_batch()
        elif self.mode == 'test':
            res = self._prepare_test_batch()
        else:
            raise NotImplementedError

        return res
    def __len__(self):
        return self.total_len

    next = __next__

def encoded_images_sizes(jpegs):
    shapes = fn.peek_image_shape(jpegs)  # the shapes are HWC
    h = fn.slice(shapes, 0, 1, axes=[0]) # extract height...
    w = fn.slice(shapes, 1, 1, axes=[0]) # ...and width...
    return fn.cat(w, h)               # ...and concatenate

def ExternalSourceTrainPipeline(batch_size, num_threads, device_id, external_data, train_width, train_height, top_crop, normalize_image_scale = False, nscale_w = None, nscale_h = None):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, seg_images, labels = fn.external_source(source=external_data, num_outputs=3)
        # todo : mixed report error, 使用cpu编解码
        images = fn.decoders.image(jpegs, device="mixed")
        seg_images = fn.decoders.image(seg_images, device="mixed")
        if normalize_image_scale:
            images = fn.resize(images, resize_x=nscale_w, resize_y=nscale_h)
            seg_images = fn.resize(seg_images, resize_x=nscale_w, resize_y=nscale_h, interp_type=types.INTERP_NN)
            # make all images at the same size

        size = encoded_images_sizes(jpegs)
        center = size / 2

        mt = fn.transforms.scale(scale = fn.random.uniform(range=(0.8, 1.2), shape=[2]), center = center)
        mt = fn.transforms.rotation(mt, angle = fn.random.uniform(range=(-6, 6)), center = center)

        off = fn.cat(fn.random.uniform(range=(-200, 200), shape = [1]), fn.random.uniform(range=(-100, 100), shape = [1]))
        mt = fn.transforms.translation(mt, offset = off)

        images = fn.warp_affine(images, matrix = mt, fill_value=0, inverse_map=False)
        seg_images = fn.warp_affine(seg_images, matrix = mt, fill_value=0, inverse_map=False)
        labels = fn.coord_transform(labels.gpu(), MT = mt)


        images = fn.resize(images, resize_x=train_width, resize_y=int(train_height/top_crop))
        seg_images = fn.resize(seg_images, resize_x=train_width, resize_y=int(train_height/top_crop), interp_type=types.INTERP_NN)


        images = fn.crop_mirror_normalize(images, 
                                            dtype=types.FLOAT, 
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        seg_images = fn.crop_mirror_normalize(seg_images, 
                                            dtype=types.FLOAT, 
                                            mean = [0., 0., 0.],
                                            std = [1., 1., 1.],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        pipe.set_outputs(images, seg_images, labels)
    return pipe

def ExternalSourceValPipeline(batch_size, num_threads, device_id, external_data, train_width, train_height):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, labels = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(jpegs, device="mixed")
        images = fn.resize(images, resize_x=train_width, resize_y=int(train_height/0.6)+1)
        images = fn.crop_mirror_normalize(images, 
                                            dtype=types.FLOAT, 
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255],
                                            crop = (train_height, train_width), crop_pos_x = 0., crop_pos_y = 1.)
        pipe.set_outputs(images, labels.gpu())
    return pipe

def ExternalSourceTestPipeline(batch_size, num_threads, device_id, external_data):
    pipe = Pipeline(batch_size, num_threads, device_id)
    with pipe:
        jpegs, names = fn.external_source(source=external_data, num_outputs=2)
        images = fn.decoders.image(jpegs, device="mixed")

        images = fn.resize(images, resize_x=800, resize_y=288)
        images = fn.crop_mirror_normalize(images, 
                                            dtype=types.FLOAT, 
                                            mean = [0.485 * 255, 0.456 * 255, 0.406 * 255],
                                            std = [0.229 * 255, 0.224 * 255, 0.225 * 255])

        names = fn.pad(names, axes=0, fill_value = -1, shape = 46)
        pipe.set_outputs(images, names)
    return pipe
# from data.constant import culane_row_anchor, culane_col_anchor
class TrainCollect:
    def __init__(self, batch_size, num_threads, data_root, list_path, shard_id, num_shards,
                 row_anchor, col_anchor, train_width, train_height, num_cell_row, num_cell_col,
                 dataset_name, top_crop):
        """
        车道线检测数据加载器
        核心功能：
        1. 使用DALI构建高效数据管道
        2. 处理不同数据集的尺寸差异
        3. 生成位置预测的离散化标签
        4. 车道线坐标插值与扩展

        参数说明：
        row_anchor/col_anchor : 预定义的行/列采样位置（归一化坐标）
        num_cell_row/col : 网格划分数量（对应模型输出维度）
        top_crop : 图像顶部裁剪像素数（去除天空等无关区域）
        """

        # 初始化数据迭代器 ------------------------------------------------------
        # LaneExternalIterator: 自定义数据加载器，读取原始标注文件
        eii = LaneExternalIterator(
            data_root,
            list_path,
            batch_size=batch_size,
            shard_id=shard_id,  # 分布式训练分片ID
            num_shards=num_shards,  # 总分布式分片数
            dataset_name=dataset_name
        )

        # 设置原始图像尺寸（不同数据集不同）---------------------------------------
        self._set_original_dims(dataset_name)

        # 构建DALI数据管道 ------------------------------------------------------
        # 注：CurveLanes需要特殊处理尺寸（2560x1440 → 缩放后保持宽高比）
        if dataset_name == 'CurveLanes':
            pipe = ExternalSourceTrainPipeline(
                batch_size, num_threads, shard_id, eii,
                train_width, train_height, top_crop,
                normalize_image_scale=True,  # 启用比例保持缩放
                nscale_w=2560, nscale_h=1440  # 原始尺寸基准
            )
        else:
            pipe = ExternalSourceTrainPipeline(
                batch_size, num_threads, shard_id, eii,
                train_width, train_height, top_crop
            )

        # 创建DALI迭代器 --------------------------------------------------------
        self.pii = DALIGenericIterator(
            pipe,
            output_map=['images', 'seg_images', 'points'],  # 输出字段映射
            last_batch_padded=True,  # 最后一个批次允许填充
            last_batch_policy=LastBatchPolicy.PARTIAL  # 部分批次处理策略
        )

        # 元数据存储 ------------------------------------------------------------
        self.eii_n = eii.n  # 数据集样本总数
        self.batch_size = batch_size

        # 坐标转换参数初始化 -----------------------------------------------------
        # 将归一化锚点转换为实际像素坐标（设备转移至GPU）
        self.interp_loc_row = torch.tensor(row_anchor, dtype=torch.float32).cuda() * self.original_image_height
        self.interp_loc_col = torch.tensor(col_anchor, dtype=torch.float32).cuda() * self.original_image_width

        # 网格参数存储
        self.num_cell_row = num_cell_row  # 行方向网格数（如72）
        self.num_cell_col = num_cell_col  # 列方向网格数（如81）

    def _set_original_dims(self, dataset_name):
        """根据数据集名称设置原始图像尺寸"""
        if dataset_name == 'CULane':
            self.original_image_width = 1640
            self.original_image_height = 590
        elif dataset_name == 'Tusimple':
            self.original_image_width = 1280
            self.original_image_height = 720
        elif dataset_name == 'CurveLanes':
            self.original_image_width = 2560
            self.original_image_height = 1440

    def __iter__(self):
        """返回迭代器自身"""
        return self

    def __next__(self):
        """获取下一个批次数据"""
        data = next(self.pii)  # 从DALI迭代器获取原始数据

        # 数据解包 -------------------------------------------------------------
        images = data[0]['images']  # 增强后的图像 [B,3,H,W]
        seg_images = data[0]['seg_images']  # 分割标签图 [B,H,W]
        points = data[0]['points']  # 原始车道线坐标 [B, num_lanes, num_points, 2]

        # 行方向标签处理 --------------------------------------------------------
        # 沿行锚点进行插值 → [B, num_lanes, num_row_anchor, 1]
        points_row = my_interp.run(points, self.interp_loc_row, 0)

        # 坐标扩展（多项式拟合补全不完整车道线）→ [B, num_lanes, num_row_anchor]
        points_row_extend = self._extend(points_row[:, :, :, 0]).transpose(1, 2)

        # 生成离散化标签 --------------------------------------------------------
        # 将X坐标映射到网格分类空间 [0, num_cell_row-1]
        labels_row = (points_row_extend / self.original_image_width * (self.num_cell_row - 1)).long()

        # 无效值处理（超出图像范围的位置标记为-1）
        labels_row[points_row_extend < 0] = -1
        labels_row[points_row_extend > self.original_image_width] = -1
        labels_row[labels_row < 0] = -1
        labels_row[labels_row > (self.num_cell_row - 1)] = -1

        # 列方向标签处理（逻辑与行方向对称）---------------------------------------
        points_col = my_interp.run(points, self.interp_loc_col, 1)
        points_col = points_col[:, :, :, 1].transpose(1, 2)
        labels_col = (points_col / self.original_image_height * (self.num_cell_col - 1)).long()
        labels_col[points_col < 0] = -1
        labels_col[points_col > self.original_image_height] = -1
        labels_col[labels_col < 0] = -1
        labels_col[labels_col > (self.num_cell_col - 1)] = -1

        # 生成浮点型标签（用于存在性判断）-----------------------------------------
        labels_row_float = points_row_extend / self.original_image_width  # 归一化到[0,1]
        labels_row_float[labels_row_float < 0] = -1
        labels_row_float[labels_row_float > 1] = -1

        labels_col_float = points_col / self.original_image_height
        labels_col_float[labels_col_float < 0] = -1
        labels_col_float[labels_col_float > 1] = -1

        return {
            'images': images,  # 输入图像
            'seg_images': seg_images,  # 分割标签
            'labels_row': labels_row,  # 行方向离散标签
            'labels_col': labels_col,  # 列方向离散标签
            'labels_row_float': labels_row_float,  # 行归一化坐标（存在性判断）
            'labels_col_float': labels_col_float  # 列归一化坐标
        }

    def __len__(self):
        """计算总批次数"""
        return int((self.eii_n + self.batch_size - 1) / self.batch_size)

    def reset(self):
        """重置迭代器"""
        self.pii.reset()

    next = __next__  # 兼容Python2

    def _extend(self, coords):
        """
        车道线坐标扩展补全
        原理：对不完整的车道线标注进行多项式拟合，生成完整坐标

        参数：
        coords : [n, num_lanes, num_cls] 原始坐标（未补全）

        返回：
        fitted_coords : 补全后的坐标
        """
        n, num_lanes, num_cls = coords.shape
        coords_np = coords.cpu().numpy()
        coords_axis = np.arange(num_cls)
        fitted_coords = coords.clone()

        for i in range(n):  # 遍历批次
            for j in range(num_lanes):  # 遍历车道线
                lane = coords_np[i, j]
                if lane[-1] > 0:  # 最后一点有效则跳过（完整车道线）
                    continue

                valid = lane > 0  # 有效点掩码
                num_valid_pts = np.sum(valid)
                if num_valid_pts < 6:  # 有效点不足时跳过
                    continue

                # 使用后半段有效点进行线性拟合 --------------------------------
                # 选择后半段点：假设车道线在近处更可靠
                p = np.polyfit(
                    coords_axis[valid][num_valid_pts // 2:],
                    lane[valid][num_valid_pts // 2:],
                    deg=1  # 一阶多项式（直线拟合）
                )

                # 确定拟合起始点（最后一个有效点的位置）
                start_point = coords_axis[valid][num_valid_pts // 2]

                # 生成拟合后的坐标（从起始点到末尾）
                fitted_lane = np.polyval(p, np.arange(start_point, num_cls))

                # 更新坐标张量
                fitted_coords[i, j, start_point:] = torch.tensor(fitted_lane, device=coords.device)
        return fitted_coords

    def _extend_col(self, coords):
        """列方向扩展（暂未实现，对称设计预留）"""
        pass


class TestCollect:
    def __init__(self, batch_size, num_threads, data_root, list_path, shard_id, num_shards):
        self.batch_size = batch_size
        eii = LaneExternalIterator(data_root, list_path, batch_size=batch_size, shard_id=shard_id, num_shards=num_shards, mode = 'test')
        pipe = ExternalSourceTestPipeline(batch_size, num_threads, shard_id, eii)
        self.pii = DALIGenericIterator(pipe, output_map = ['images', 'names'], last_batch_padded=True, last_batch_policy=LastBatchPolicy.PARTIAL)
        self.eii_n = eii.n
    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.pii)
        images = data[0]['images']
        names = data[0]['names']
        restored_names = []
        for name in names:
            if name[-1] == -1:
                restored_name = ''.join(list(map(chr,name[:-1])))
            else:
                restored_name = ''.join(list(map(chr,name)))
            restored_names.append(restored_name)
            
        out_dict = {'images': images, 'names': restored_names}
        return out_dict
    
    def __len__(self):
        return int((self.eii_n + self.batch_size - 1) / self.batch_size)

    def reset(self):
        self.pii.reset()
    next = __next__


