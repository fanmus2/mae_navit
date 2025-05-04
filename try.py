import torch
import torch.nn as nn
import random
import torch.multiprocessing as mp
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


from time_folder import ImageFolder
import torch
from typing import List, Optional, Callable
from functools import partial



def default(val, d):
    return val if val is not None else d

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

import torch
import random
from typing import List, Tuple, Optional, Callable

def custom_collate_fn(
    batch: List[Tuple[torch.Tensor, int, int]],
    calc_token_dropout: Optional[Callable] = None,
    max_seq_len: int = 256
) -> Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """
    自定义时序序列打包函数，按最后一个维度合并
    Args:
        batch: 输入批次数据，每个张量形状为 (T, 1, C, L)
        calc_token_dropout: 计算 token dropout 的函数
        max_seq_len: 最大序列长度
    Returns:
        packed_batch: 打包后的批次张量，形状为 (num_samples, 1, C, max_seq_len)
        packed_labels: 对应的标签列表，每个元素是一个张量，形状为 (num_samples_in_group,)
        packed_adjusted_lengths: 对应的调整后长度列表，每个元素是一个张量，形状为 (num_samples_in_group,)
        packed_indices: 对应的索引列表，每个元素是一个张量，形状为 (num_samples_in_group,)
    """
    calc_token_dropout = calc_token_dropout or (lambda: 0.0)

    # 处理每个样本并计算调整后的长度
    processed_items = []
    for item, index, label in batch:
        orig_seq_len = item.shape[-1]  # 原始序列长度 (L)
        
        # 计算 dropout 率（根据实际需求传入参数）
        dropout_rate = calc_token_dropout()  # 示例无参数，可按需修改
        adjusted_len = int(orig_seq_len * (1 - dropout_rate))
        
        # 校验调整后的长度
        if adjusted_len > max_seq_len:
            adjusted_len = max_seq_len  # 限制最大长度
        
        # 将 (T, 1, C, L) 转换为 T 个 (1, 1, C, L)
        for t in range(item.shape[0]):
            tensor = torch.from_numpy(item[t]).unsqueeze(0)  # 增加一个维度，使其变为 (1, 1, C, L)
            processed_items.append((tensor, adjusted_len, label, index))

    # 打乱顺序
    random.shuffle(processed_items)

    # 分组逻辑
    groups = []
    current_group = []
    current_len = 0

    for tensor, adj_len, label, index in processed_items:
        if current_len + adj_len > max_seq_len:
            groups.append(current_group)
            current_group = []
            current_len = 0
        current_group.append((tensor, adj_len, label, index))
        current_len += adj_len

    if current_group:
        groups.append(current_group)

    # 修改点：为每个 group 独立保存数据和标签
    packed_batch = []
    packed_labels = []
    packed_adjusted_lengths = []
    packed_indices = []
    for group in groups:
        truncated_tensors = []
        group_labels = []
        group_adjusted_lengths = []
        group_indices = []
        for tensor, adj_len, label, index in group:
            truncated = tensor[..., :adj_len]
            truncated_tensors.append(truncated)
            group_labels.append(label)
            group_adjusted_lengths.append(adj_len)
            group_indices.append(index)
        
        # 拼接和填充
        concatenated = torch.cat(truncated_tensors, dim=-1)
        seq_len = concatenated.shape[-1]
        
        if seq_len < max_seq_len:
            pad_shape = concatenated.shape[:-1] + (max_seq_len - seq_len,)
            padding = torch.zeros(pad_shape, dtype=concatenated.dtype, device=concatenated.device)
            concatenated = torch.cat([concatenated, padding], dim=-1)
        
        packed_batch.append(concatenated)
        packed_labels.append(torch.tensor(group_labels))  # 转换为张量的一维数组
        packed_adjusted_lengths.append(torch.tensor(group_adjusted_lengths))  # 转换为张量的一维数组
        packed_indices.append(torch.tensor(group_indices))  # 转换为张量的一维数组

    # 将列表转换为张量
    packed_batch = torch.stack(packed_batch)  # 形状为 (num_samples, 1, C, max_seq_len)

    return packed_batch, packed_labels, packed_adjusted_lengths, packed_indices
if __name__ == '__main__':
    
    dataset = ImageFolder(
        root="../../datasets/data_sho_421",

    )
    print(len(dataset))
    collate_with_dropout = partial(
        custom_collate_fn,
        calc_token_dropout=None,  # 示例：固定 dropout 率为 0.1
        max_seq_len=256
    )
    dataset_loader = DataLoader(
        dataset,
        batch_size=256,  # 设置批次大小
        shuffle=True,
        num_workers=1,
        collate_fn=collate_with_dropout  # 使用自定义 collate 函数
    )
    for batch_idx, (batch_data, batch_labels, batch_adjusted_lengths, batch_indices) in enumerate(dataset_loader):
        print(f"Batch {batch_idx}:")
        print(f"  Data shape: {batch_data.shape}")
        print(f"  Labels: {batch_labels}")
        print(f"  Adjusted lengths: {batch_adjusted_lengths}")
        print(f"  Indices: {batch_indices}")
