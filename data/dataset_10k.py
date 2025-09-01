
import torch
import numpy as np
import os
import json
import random
import torchvision.transforms as transforms

from .data_utils.dataset_10k_parse import Dataset10KParse

class Dataset10K(torch.utils.data.Dataset):
    def __init__(
        self,
        data_root=None,
        video_sample_stride=4, video_sample_n_frames=16,
        text_drop_ratio=0.1,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        sample_size=None
    ):  
        self.data_root = data_root
        json_file_path = os.path.join(data_root, "metadata.json")
        self.dataset = json.load(open(json_file_path))                           # 加载json文件

        print("Load Dataset From:", json_file_path)

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")                         # 数据集大小

        self.text_drop_ratio = text_drop_ratio

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride        # 采样间隔 3
        self.video_sample_n_frames  = video_sample_n_frames      # 期待的帧数
        self.video_transforms = transforms.Compose(
            [   
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
        self.sample_size = sample_size

    def get_batch(self, idx):
        json_data_info = self.dataset[idx % len(self.dataset)]
        dataset_dir = os.path.join(self.data_root, json_data_info['file_path'])

        scene_json_file = os.path.join(dataset_dir, "_scene_meta_backup.json")
        if not os.path.exists(scene_json_file):
            raise ValueError(f"Scene Json File Not exsit!")
        scene_info = json.load(open(scene_json_file, 'r'))
        scene_frames_info = scene_info["frames"]

        video_len = len(scene_frames_info)        # 视频长度

        # 期待帧数 self.video_sample_n_frames   (多读取2帧)
        # 现有视频有效长度 int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start),  有效帧数 = 有效长度 / 采样间隔
        # 期待帧数和有效长度直接选最小（考虑了采样间隔）
        min_sample_n_frames = min(
            self.video_sample_n_frames, 
            int(video_len * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
        )
        if min_sample_n_frames == 0:
            raise ValueError(f"No Frames in video.")

        video_length = int(self.video_length_drop_end * video_len)    # 视频有效帧的末尾位置    324
        clip_length = min(video_length, (min_sample_n_frames - 1) * self.video_sample_stride + 1)    # 最多采样帧数 * 时间跨度 = 最远采样长度

        # 随机选择一个起始帧 & 生成要抽取帧的索引
        start_idx   = random.randint(int(self.video_length_drop_start * video_length), video_length - clip_length) if video_length != clip_length else 0
        batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

        dataset_10k_parse_tools = Dataset10KParse(dataset_dir)
        data_info = dataset_10k_parse_tools.parse(batch_index, self.sample_size)
        del dataset_10k_parse_tools

        final_info = {}
        if data_info == None:
            print("Parse File Failed:", dataset_dir)
            raise ValueError(f"Parse Error, Check you dataset {dataset_dir}")
        else:
            start_c2w, end_c2w = data_info["c2w"][0].copy(), data_info["c2w"][-1].copy()
            start_w2c, end_w2c = data_info["w2c"][0].copy(), data_info["w2c"][-1].copy()
            start_intrinsic, end_intrinsic = data_info["intrinsics"][0].copy(), data_info["intrinsics"][-1].copy()
            start_depth, end_depth = data_info["depth"][0].copy(), data_info["depth"][-1].copy()

            cache_depth = np.expand_dims(np.stack([start_depth, end_depth], axis=0), axis=1)
            cache_intrinsic = np.stack([start_intrinsic, end_intrinsic], axis=0)  # (B, 3, 3)
            cache_w2c = np.stack([start_w2c, end_w2c], axis=0)    # (B, 4, 4)

            pixel_values = data_info["frames"]
            target_w2c = data_info["w2c"]
            target_intrinsic = data_info["intrinsics"]

            pixel_values = torch.from_numpy(pixel_values).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.
            pixel_values = self.video_transforms(pixel_values)    # -1, 1之间

            cache_frames = torch.stack([pixel_values[0], pixel_values[-1]], dim=0)   # 用于cache_3d的构建

            # 去掉首尾的视频
            final_info["pixel_values"] = pixel_values
            final_info["target_intrinsic"] = target_intrinsic
            final_info["target_w2c"] = target_w2c
            # 用于构建cache_3d的信息
            final_info["cache_frames"] = cache_frames
            final_info["cache_depth"] = cache_depth
            final_info["cache_intrinsic"] = cache_intrinsic
            final_info["cache_w2c"] = cache_w2c

        text = json_data_info.get('text', '')
        if random.random() < self.text_drop_ratio:
            text = ''

        return final_info, text, 'video', dataset_dir

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            sample = {}
            try:
                data_info, text, file_type, dataset_dir = self.get_batch(idx)
                sample = data_info
                sample["text"] = text
                sample["data_type"] = file_type
                sample["idx"] = idx
                sample["file_name"] = dataset_dir
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)
        return sample