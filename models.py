import torch
import torch.nn as nn
import numpy as np
from einops import rearrange
import torch.nn.functional as F
from functools import partial
from timm.models.vision_transformer import Block
from typing import  Optional ,Type
from typing import  Optional
from timm.layers import  Mlp, DropPath
from torch.jit import Final
from einops import rearrange, repeat
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, \
    OPENAI_CLIP_MEAN, OPENAI_CLIP_STD
from timm.layers import PatchEmbed, Mlp, DropPath, AttentionPoolLatent, RmsNorm, PatchDropout, SwiGLUPacked, SwiGLU, \
    trunc_normal_, lecun_normal_, resample_patch_embed, resample_abs_pos_embed, use_fused_attn, \
    get_act_layer, get_norm_layer, LayerType
def exists(val):
    return val is not None   
def default(val, d):
    return val if exists(val) else d

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

#they use a query-key normalization that is equivalent to rms norm (no mean-centering, learned gamma), from vit 22B paper
class RMSNorm(nn.Module):
    def __init__(self, heads, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(heads, 1, dim))

    def forward(self, x):
        normed = F.normalize(x, dim = -1)
        return normed * self.scale * self.gamma

#feedforward
class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,           # 输入维度
        hidden_dim: int,   # 隐层维度
        dropout: float = 0. # Dropout概率
    ):
        super().__init__()
        # 定义网络结构（与之前函数中的nn.Sequential一致）
        self.net = nn.Sequential(
            LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # 必须实现forward方法

      
class Attention(nn.Module):
    fused_attn: Final[bool]

    def __init__(
            self,
            dim: int,
            num_heads: int = 8,
            qkv_bias: bool =False,
            qk_norm: bool = True,
            proj_bias: bool = True,
            attn_drop: float = 0.3,
            proj_drop: float = 0.3,
            norm_layer=RMSNorm,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.norm = LayerNorm(dim)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = RMSNorm(self.num_heads,dim=self.head_dim) 
        self.k_norm = RMSNorm(heads=self.num_heads,dim=self.head_dim) 
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.to_q = nn.Linear(dim, dim, bias = qkv_bias)
        self.to_kv = nn.Linear(dim, dim * 2, bias = qkv_bias)
    def forward(self, x: torch.Tensor,attn_mask: Optional[torch.Tensor] = None,attn_mask_fintune: Optional[torch.Tensor] = None,context=None) -> torch.Tensor:
        x = self.norm(x)
        kv_input = default(context, x)
        B, N, C = x.shape
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        qkv = (self.to_q(x), *self.to_kv(kv_input).chunk(2, dim = -1))
        q, k, v = map(
            lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.num_heads), qkv
        )
        # q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
    
        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
                    # 应用布尔型 attn_mask（关键修改）
            if attn_mask is not None:
                original_n = attn_mask.shape[-1]
                new_n = original_n + 1
                # 创建一个新的掩码，形状为 (N, 1, new_n, new_n) 并且所有值都为 True
                new_attn_mask = torch.full((B, 1, new_n, new_n), True, dtype=torch.bool, device=attn_mask.device)                
                # 如果需要保留原始掩码的部分，可以直接复制过来
                new_attn_mask[..., 1:, 1:] = attn_mask
                # 替换原始的掩码
                attn_mask = new_attn_mask
                attn = attn.masked_fill(~attn_mask, float('-inf'))  # ~ 表示逻辑非
                
            if attn_mask_fintune is not None:
                original_n = attn_mask_fintune.shape[-1]
                new_n = original_n + 1
                # 创建一个新的掩码，形状为 (N, 1, new_n, new_n) 并且所有值都为 True
                new_attn_mask_fintune = torch.full((B, 1, attn_mask_fintune.shape[-2], new_n), True, dtype=torch.bool, device=attn_mask_fintune.device)                
                # 如果需要保留原始掩码的部分，可以直接复制过来
                new_attn_mask_fintune[..., 1:] = attn_mask_fintune
                # 替换原始的掩码
                attn_mask_fintune = new_attn_mask_fintune
                attn = attn.masked_fill(~attn_mask_fintune, float('-inf'))  # ~ 表示逻辑非    

            attn = attn.softmax(dim=-1)
            attn=torch.where(torch.isnan(attn),torch.full_like(attn,0),attn)
            attn = self.attn_drop(attn)
            x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# class LayerScale(nn.Module):
#     def __init__(
#             self,
#             dim: int,
#             init_values: float = 1e-5,
#             inplace: bool = False,
#     ) -> None:
#         super().__init__()
#         self.inplace = inplace
#         self.gamma = nn.Parameter(init_values * torch.ones(dim))

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         return x.mul_(self.gamma) if self.inplace else x * self.gamma

    
class Block_change(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_bias: bool = True,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: Type[nn.Module] = nn.GELU,
            norm_layer: Type[nn.Module] = RMSNorm,
            mlp_layer: Type[nn.Module] = FeedForward,
    ) -> None:
        super().__init__()
        self.norm1 = LayerNorm(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        # self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = LayerNorm(dim)
        self.mlp = mlp_layer(
            dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            dropout=proj_drop,
        )
        # self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor,attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        print(torch.cuda.memory_allocated(3)/(1024**2),"MB1,BLOCK1")
        # x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x),attn_mask=attn_mask)))  
        # x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = x + self.drop_path1(self.attn(self.norm1(x),attn_mask=attn_mask))  
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        print(torch.cuda.memory_allocated(3)/(1024**2),"MB1,BLOCK2")
        return x



class STMAE_Pre(nn.Module):
    def __init__(self, embed_dim=3, depth=10, num_heads=4,
                 decoder_embed_dim=3, decoder_depth=2, decoder_num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 node_dim=6, window_size=512, node_num=7, mask_ratio=0.5, len_mask=1,proj_drop=0,attn_drop=0,in_out_dim=3):
        super().__init__()

        self.len_mask = len_mask
        self.mask_ratio = mask_ratio
        self.window_size = window_size
        self.node_num = node_num
        # self.conv1 = nn.Conv2d(node_dim, embed_dim, kernel_size=(self.node_num, 5), stride=1, padding=(0, 2))
        self.conv1 = nn.Conv2d(3, embed_dim, kernel_size=(1, 3), stride=1, padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size + 1, embed_dim), requires_grad=False)
        # self.blocks = nn.ModuleList([
        #     Block_change(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,proj_drop=proj_drop,attn_drop=attn_drop)
        #     for i in range(depth)])
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, window_size + 1, decoder_embed_dim), requires_grad=False)     
        # self.decoder_blocks = nn.ModuleList([
        #     Block_change(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,proj_drop=proj_drop,attn_drop=attn_drop)
        #     for i in range(decoder_depth)])
        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])
        self.decoder_norm = norm_layer(decoder_embed_dim)
        # self.decoder_pred = nn.Linear(decoder_embed_dim, node_dim*node_num, bias=True)
        self.decoder_pred = nn.Linear(decoder_embed_dim,in_out_dim, bias=True)
        self.initialize_weights()

    def initialize_weights(self):
        pos_embed = get_ts_sincos_pos_embed(self.pos_embed.shape[-1], int(self.window_size), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        decoder_pos_embed = get_ts_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.window_size), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))
        w = self.conv1.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio,attn_mask,windows_len):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        mask = torch.ones(N, L)  # 初始化全遮蔽    
        # 为每个样本的窗口生成遮蔽
        for i in range(N):
            sum_len=0
            start = 0
            for win_len in windows_len[i]:
                win_len = win_len.item()
                end = start + win_len            
                # 计算当前窗口需要保留的token数
                keep = int(win_len * (1 - mask_ratio))
                sum_len+=keep
                # 生成窗口内的随机排列
                perm = torch.randperm(win_len)
                keep_indices = perm[:keep]  # 窗口内保留的位置
                # 将保留位置的mask设为0
                global_indices = start + keep_indices
                mask[i, global_indices] = 0      
                start = end
            if len_keep-sum_len>0:
                mask[i, -(len_keep-sum_len):] = 0

        # sort noise for each sample
        ids_shuffle = torch.argsort(mask, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1).to(x.device)   
        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep].to(x.device)
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        if attn_mask is not None:
            ids_keep_expanded = ids_keep.view(N, 1, len_keep, 1).expand(-1, 1, -1, L)
    # 提取保留行
            attn_mask_rows = torch.gather(attn_mask, dim=2, index=ids_keep_expanded)        
            # 提取保留列
            ids_keep_expanded_col = ids_keep.view(N, 1, 1, len_keep).expand(-1, 1, len_keep, -1)
            attn_mask_masked = torch.gather(attn_mask_rows, dim=3, index=ids_keep_expanded_col)
        else:
            attn_mask_masked = None
        return x_masked, mask, ids_restore,attn_mask_masked

    
    def forward_encoder(self,x,attn_mask,batch_adjusted_lengths, mask_ratio):
        x = x + self.pos_embed[:, 1:x.shape[1]+1, :]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore,attn_mask_masked = self.random_masking(x, mask_ratio,attn_mask,batch_adjusted_lengths)
        attn_mask_masked=attn_mask_masked.to(x.device)
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x, mask, ids_restore
    
    def forward_decoder(self, x,ids_restore,attn_mask):
        # embed tokens
        x = self.decoder_embed(x)
        # append mask tokens to sequence

        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1) 
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        x = torch.cat([x[:, :1, :], x_], dim=1)
        # add pos embed
        x = x + self.decoder_pos_embed[:, 0:x.shape[1]+1, :]
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]
        

        return x

    def forward(self, imgs,attn_mask,batch_adjusted_lengths): 
        # imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], self.node_num, -1)
        imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], 1,-1)
        imgs = imgs.permute(0, 2, 1 ,3)

        # mask = torch.zeros(imgs.shape[0], imgs.shape[1], dtype=torch.bool).to(imgs.device)
        # noise = torch.rand(imgs.shape[0], imgs.shape[1]).to(imgs.device)
        # _, indices = torch.topk(noise, self.len_mask, dim=1)
        # mask.scatter_(1, indices, True)
        # mask_expanded = mask.unsqueeze(-1).unsqueeze(-1).expand_as(imgs)
        # imgs[mask_expanded] = 0

        x = imgs.permute(0, 3, 1, 2)
        x = self.bn1(self.conv1(x))
        x = x.squeeze()
        x = x.permute(0, 2, 1)
        latent, mask, ids_restore = self.forward_encoder(x, attn_mask,batch_adjusted_lengths,mask_ratio=self.mask_ratio)
        pred = self.forward_decoder(latent, ids_restore,attn_mask)
        imgs = imgs.permute(0, 2, 1, 3)
        imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], -1)

        idx = mask.nonzero()
        pred = pred[idx[:, 0], idx[:, 1], :]
        imgs = imgs[idx[:, 0], idx[:, 1], :]
        pred = pred.reshape(x.shape[0], -1, pred.shape[1])
        imgs = imgs.reshape(x.shape[0], -1, imgs.shape[1])
        return imgs, pred

class STMAE_Finetune(nn.Module):
    def __init__(self, embed_dim=512, depth=6, num_heads=4,
                 mlp_ratio=4., norm_layer=nn.LayerNorm,
                 node_dim=6, window_size=150, node_num=7, num_classes=8):
        super().__init__()
        
        self.head = nn.Linear(embed_dim, num_classes)

        self.window_size = window_size
        self.node_num = node_num
        self.conv1 = nn.Conv2d(node_dim, embed_dim, kernel_size=(self.node_num, 3), stride=1, padding=(0, 1))
        self.bn1 = nn.BatchNorm2d(embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size + 1, embed_dim), requires_grad=False)
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.attn_pool_queries = nn.Parameter(torch.randn(embed_dim))
        self.attn_pool = Attention(embed_dim, num_heads = 2)
        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_ts_sincos_pos_embed(self.pos_embed.shape[-1], int(self.window_size), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.conv1.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder_full(self, x):

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:x.shape[1]+1, :]
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x

    def forward(self, imgs,num_images,batched_image_ids,key_pad_mask):
        
        imgs = imgs.reshape(imgs.shape[0], imgs.shape[1], self.node_num, -1)
        imgs = imgs.permute(0, 2, 1 ,3) 
        x = imgs.permute(0, 3, 1, 2)
        arange = partial(torch.arange, device = x.device)
        num_images=num_images.to(x.device)
        batched_image_ids=batched_image_ids.to(x.device)
        key_pad_mask=key_pad_mask.to(x.device)
        x = self.bn1(self.conv1(x))
        x = x.squeeze()
        x = x.permute(0, 2, 1)
        x = self.forward_encoder_full(x)
        max_queries = num_images.amax().item()
        queries = repeat(self.attn_pool_queries, 'd -> b n d', n = max_queries, b = x.shape[0])
        #attention pool mask
        image_id_arange = arange(max_queries)
        attn_pool_mask = rearrange(image_id_arange, 'i -> i 1') == rearrange(batched_image_ids, 'b j -> b 1 j')
        attn_pool_mask = attn_pool_mask & rearrange(key_pad_mask, 'b j -> b 1 j')
        attn_pool_mask = rearrange(attn_pool_mask, 'b i j -> b 1 i j')
        #attention pool
        x = self.attn_pool(queries, attn_mask_fintune = attn_pool_mask, context = x) + queries
        x = rearrange(x, 'b n d -> (b n) d')
     #each batch element may not have same amount of images
        is_images = image_id_arange < rearrange(num_images, 'b -> b 1')
        is_images = rearrange(is_images, 'b n -> (b n)')
        x = x[is_images]
        x = self.head(x)

        return x
    
def get_ts_sincos_pos_embed(embed_dim, window_size, cls_token=False):
    """
    embed_dim: output dimension for each position
    window_size: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    # assert embed_dim % 2 == 0
    # 注释了！！！
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = np.arange(window_size, dtype=np.float32)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    pos_embed = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0) # (M+1, D)
    return pos_embed
    
def fetch_classifier(method, args=None):
    if 'STMAE_Pre' in method:
        model = STMAE_Pre(embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, 
        norm_layer=nn.LayerNorm, node_dim=3, window_size=args.maxlen, node_num=1,
        decoder_embed_dim=args.decoder_embed_dim, decoder_depth=args.decoder_depth, decoder_num_heads=args.decoder_num_heads, 
        mask_ratio=args.mask_ratio, len_mask=args.len_mask,proj_drop=args.proj_drop,attn_drop=args.attn_drop,in_out_dim=args.in_out_dim)
    elif 'STMAE_Finetune' in method:
        model = STMAE_Finetune(embed_dim=args.embed_dim, depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio,
        norm_layer=nn.LayerNorm, node_dim=3, window_size=args.maxlen, node_num=1, num_classes=args.dataset_cfg.activity_label_size)
    else:
        model = None
    return model
