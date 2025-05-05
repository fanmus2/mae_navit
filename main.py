import os
import torch
import torch.nn as nn
from einops import rearrange, repeat
# from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import random
import time
from torch.utils.data import DataLoader
from time_folder import ImageFolder
from typing import List, Optional, Callable
from functools import partial
from utils import *
from models import fetch_classifier
import train
from typing import List, Tuple, Optional, Callable
from torch.nn.utils.rnn import pad_sequence 

def custom_collate_fn(
    batch: List[Tuple[np.array, int, int]],
    calc_token_dropout: Optional[Callable] = None,
    max_seq_len: int = 1024
) -> Tuple[List[torch.tensor], List[torch.tensor], List[torch.tensor], List[torch.tensor],List[torch.tensor]]:

        # 处理每个样本并计算调整后的长度
    processed_items = []
    for item, index, label in batch:
        adjusted_len= item.shape[-1]  # 原始序列长度 (L)
        # 将 (T, 1, C, L) 转换为 T 个 (1, 1, C, L)
        for t in range(item.shape[0]):
            processed_items.append((item[t], adjusted_len, label, index))        
    random.shuffle(processed_items)         
    packed_batch = []
    packed_labels = []
    packed_adjusted_lengths = []
    packed_indices = []
    batched_image_ids=[]
    
    buffer = np.empty((3, max_seq_len), dtype=np.float32)
    labels=[]
    lengths=[]
    image_ids=[]
    indices=[]
    ptr = 0 
    id=0# 当前写入位置指针
    
    for item,adjusted_len_one_channel ,label,index in processed_items:
        #  计算合并后的总长度 (n * adjusted_len_one_channel)
        C = item.shape[1]  # 通道数（注意：应在拆分前获取）
        assert C % 3 == 0, f"通道数必须是3的倍数，当前为{C}."
        n = C // 3
        adjusted_len = n * adjusted_len_one_channel
        adjusted_len = min(adjusted_len, max_seq_len)  # 限制总长度   
        # --- 数据处理与合并 ---     
        # 拆分并合并通道
        item = item.reshape(1, n, 3, adjusted_len_one_channel)
        item = np.transpose(item, (0, 2, 1, 3))  # (1, 3, n, adjusted_len_one_channel)
        item = item.reshape(1, 3, -1)  # (1, 3, adjusted_len)
        item=item[:,:,0:adjusted_len]
        item = np.squeeze(item, axis=0)  #(3, adjusted_len)
        sub_len = adjusted_len   
        # 在打包逻辑中:
        if ptr + sub_len > max_seq_len:
            if ptr > 0:
                # 截取有效数据
                concatenated = buffer[:, :ptr] 
                concatenated =torch.from_numpy(concatenated)
                packed_batch.append(concatenated.permute(1,0))
                packed_labels.append(torch.tensor(labels))
                packed_adjusted_lengths.append(torch.tensor(lengths))#在最开始用extend会不会好点？
                packed_indices.append(torch.tensor(indices))  
                batched_image_ids.append(torch.tensor(image_ids))     
                # 重置指针
                ptr = 0
                id=0
                labels=[]
                indices=[]
                lengths=[]
                image_ids=[]
        # 添加新元素时:
        if ptr + sub_len <=max_seq_len:
            id=id+1
            buffer[ :, ptr:ptr+sub_len] = item
            ptr += sub_len
            labels.append(label)
            lengths.append(sub_len)
            indices.append(index)
            image_ids.extend([id] * sub_len)

    return packed_batch,packed_labels, packed_adjusted_lengths, packed_indices,batched_image_ids



def pre_train(args, data_train):
    collate_with_dropout = partial(
        custom_collate_fn,
        calc_token_dropout=None,  # 示例：固定 dropout 率为 0.1
        max_seq_len=args.maxlen
    )
    data_set_train = DataLoader(
        data_train,
        batch_size=args.batch_size,  # 设置批次大小
        shuffle=True,
        num_workers=12,
        collate_fn=collate_with_dropout,
        pin_memory=True# 使用自定义 collate 函数
    )
    criterion = nn.MSELoss(reduction='none')
    model = fetch_classifier('STMAE_Pre', args=args)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.pre_lr)
    trainer = train.Trainer(model, optimizer, args.save_path, get_device(args.gpu), args)

    def func_loss(model, batch,mask,batch_adjusted_lengths):
        # data, _ = batch
        data=batch
        seqs, seq_recon = model(data,mask, batch_adjusted_lengths)
        loss = criterion(seq_recon, seqs)
        return loss

    def func_forward(model, batch):
        # data, _ = batch
        data=batch
        seqs, seq_recon = model(data)
        return seq_recon, seqs

    def func_evaluate(seqs, predict_seqs):
        loss_lm = criterion(predict_seqs, seqs)
        return loss_lm.mean().cpu().numpy()

    log_path = os.path.join('check', args.dataset, args.path)
    writer = SummaryWriter(log_path)

    trainer.pretrain(func_loss, func_forward, func_evaluate, data_set_train,
                     model_file=args.pretrain_model, writer=writer)

def fine_tuning(args, data_train_l, label_train_l, data_valid, label_valid, data_test, label_test):
    data_set_train = IMUDataset(data_train_l, label_train_l)
    data_set_valid = IMUDataset(data_valid, label_valid)
    data_set_test = IMUDataset(data_test, label_test)
    data_loader_train = DataLoader(data_set_train, shuffle=True, batch_size=args.batch_size)
    data_loader_valid = DataLoader(data_set_valid, shuffle=False, batch_size=args.batch_size)
    data_loader_test = DataLoader(data_set_test, shuffle=False, batch_size=args.batch_size)

    criterion = nn.CrossEntropyLoss()
    model = fetch_classifier("STMAE_Finetune", args=args)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.fine_lr)
    trainer = train.Trainer(model, optimizer, args.save_path, get_device(args.gpu), args)

    def func_loss(model, batch):
        inputs, label = batch
        logits = model(inputs)
        loss = criterion(logits, label)
        return loss

    def func_forward(model, batch):
        inputs, label = batch
        logits = model(inputs)
        return logits, label

    def func_evaluate(label, predicts):
        stat = stat_acc_f1(label.cpu().numpy(), predicts.cpu().numpy())
        return stat

    log_path = os.path.join('check', args.dataset, args.path)
    writer = SummaryWriter(log_path)

    trainer.fine_tuning(func_loss, func_forward, func_evaluate, data_loader_train, data_loader_valid, data_loader_test, 
                  model_file=args.pretrain_model, writer=writer)

if __name__ == "__main__":
    args = handle_argv_pre_train()
    # data_train_u, label_train_u, data_train_l, label_train_l, data_valid, label_valid, data_test, label_test = load_data(args)
    data_train = ImageFolder(
        root="../data/data/data_sho_421",
)
    print("start pre-train\n")
    pre_train(args, data_train)

    # print("start fine-tuning\n")
    # args = handle_argv_finetune(args)
    # fine_tuning(args, data_train_l, label_train_l, data_valid, label_valid, data_test, label_test)

    # print("dataset:{}, test_user: {}, path: {}".format(args.dataset, args.test_user, args.path))

    