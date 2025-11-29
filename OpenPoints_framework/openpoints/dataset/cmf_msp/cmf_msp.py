import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from ..build import DATASETS  # 若不是openpoints官方，可直接注释掉注册器相关内容

@DATASETS.register_module()  # 仅当使用 openpoints 注册机制时保留
class MSPDataset(Dataset):
    def __init__(self, data_root, split, split_file=None,
                 transform=None, voxel_size=None, variable=False,
                 loop=1, shuffle=True, num_points=8192, **kwargs):
        self.data_root = data_root
        self.split = split
        self.transform = transform
        self.voxel_size = voxel_size
        self.variable = variable
        self.loop = loop
        self.shuffle = shuffle
        self.num_points = num_points

        # 读取分割的JSON文件（train/val/test）
        assert split_file is not None, "split_file must be provided"
        with open(split_file, 'r') as f:
            self.patient_ids = json.load(f)

        self.data_list = [os.path.join(data_root, pid, 'point_label.txt') for pid in self.patient_ids]
        self.num_classes = 2
        self.classes = ['negative', 'positive']  # 你可以替换为有意义的名字
        self.cmap = [[255, 0, 0], [0, 255, 0]]  # 类别可视化颜色
        self.num_per_class = None  # 如需平衡loss，可预先计算后填入

    def __len__(self):
        return len(self.data_list) * self.loop

    def __getitem__(self, idx):
        idx = idx % len(self.data_list)
        file_path = self.data_list[idx]
        data = np.loadtxt(file_path)
        coord = data[:, :3].astype(np.float32)
        feat = data[:, 3:6].astype(np.float32) / 255.
        label = data[:, 6].astype(np.int64)

        coord -= coord.min(0)

        # ===> 固定随机采样 8192 个点
        num_points = self.num_points
        N = coord.shape[0]
        if N >= num_points:
            choice = np.random.choice(N, num_points, replace=False)
        else:
            choice = np.random.choice(N, num_points, replace=True)

        coord = coord[choice]
        feat = feat[choice]
        label = label[choice]

        # sample = {
        #     'pos': coord,
        #     'x': np.concatenate([coord, feat], axis=1),  # XYZ + RGB → 6通道
        #     'y': label
        # }

        sample = {
            'pos': coord,
            'x': coord,  # XYZ + RGB → 6通道
            'y': label
        }


        if self.transform:
            sample = self.transform(sample)

        if 'heights' not in sample:
            sample['heights'] = torch.from_numpy(coord[:, 2:3].astype(np.float32))

        return sample
