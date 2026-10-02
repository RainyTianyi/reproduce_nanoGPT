import os
import math
import time
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

from hellaswag import render_example, iterate_examples

# 模型参数，使用 dataclass 自动生成类的基本函数，如构造函数等等
@dataclass
class GPTConfig:
    block_size: int = 1024  # 最大序列长度
    vocab_size: int = 50257 # 词表大小
    n_layer: int = 12   # 解码器层数
    n_head: int = 12    # 多头注意力
    n_embd: int = 768   # 嵌入向量长度，也是 Transformer 隐藏层维度

# 多头注意力
class CausalSelfAttention(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # 嵌入向量必须整除头数
        assert config.n_embd % config.n_head == 0
        # K Q V 输入投影层
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # V 输出投影层
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # "bias" 用于 masked 注意力权重（解码器需要屏蔽序列后方信息）
        # 使用 register_buffer 注册模型张量，但不可学习
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .reshape(1, 1, config.block_size, config.block_size))
        # 记录 head 和 embd 数量，用于 forward 时做多头注意力
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # 注意力层输出后就是 res，所以需要使用 scale 初始化
        self.c_proj.NANOGPT_SCALE_INIT = 1
        
    def forward(self, x):
        # 从输入读取信息：batch_size, seq_len, n_embd
        B, T, C = x.size()
        qkv = self.c_attn(x)
        # 分割出 Q K V
        q, k, v = qkv.split(self.n_embd, dim=2)
        # 分割出多头，并把头放到第二维 交换后（batch_size, n_head, seq_len, n_embd // n_head）
        # 这样在计算注意力权重和加权平均时，可以使用 @ 操作（就是 bmm 乘法）
        k = k.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.reshape(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        
        """使用传统注意力
        # 缩放点积计算注意力权重
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 使用掩码 maseked_fill(条件，True时用以覆盖的值)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # Softmax 后得到注意力权重
        att = F.softmax(att, dim=-1)
        # 计算加权平均值
        y = att @ v
        """
        # 使用 torch 实现的 Flash Attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        # 恢复形状 并通过交换维度和 reshape 连接多头注意力各自的输出
        y = y.transpose(1, 2).reshape(B, T, C)
        # 对 concat 后的结果做输出投影
        y = self.c_proj(y)
        return y
        

# 用于 FFN 的全连接层
class MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh') # 使用 GELU（优化 ReLU）的 tanh 近似
        # GPT2 使用了 tanh 近似，理由是运算更快。实际上现在已经直接使用 GELU 的为多数。
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        # MLP 输出后就是 res，所以需要使用 scale 初始化
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        x = self.c_proj(self.gelu(self.c_fc(x)))
        return x
    
# Transformer 解码器块，使用掩码多头注意力，删去交叉注意力层。
class Block(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # GPT2 使用 pre-norm，即在每次进入某个运算层之前进行层归一化
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

# GPT 前向推理
class GPT(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        self.config = config    # 从 config 读入超参数
        
        # 定义核心模型，使用 ModuleDict 自定义层的名称
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),   # token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd),   # position embedding
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),   # Transformer 解码器层
            ln_f = nn.LayerNorm(config.n_embd) # GPT2 新增层归一化
        ))
        # 最后的线性层，转回词表
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # GPT 中使用了 embedding 权重共享
        # 即 token embedding 和最后的线性层共享权重，减少大量参数的同时提高模型性能
        self.transformer.wte.weight = self.lm_head.weight
        
        # 使用 GPT2 的初始化方法
        self.apply(self._init_weights)
    
    # GPT2 有自己的权重初始化方法    
    def _init_weights(self, module):
        # 对于线性层一般用 std=0.02
        # 但针对 res 之前的输出，为了稳定输出的方差，进行 scale
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                # 统计总共进行 res 的次数，进行 std scale
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                # GPT2 对 bias 项统一使用 0 初始化，而不是 torch 的均匀分布
                torch.nn.init.zeros_(module.bias)
        # 对于嵌入层同样用 std=0.02
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
    def forward(self, idx, labels=None):
        # idx (B, T)
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # 从原始的 vocab_index 转为 token embedding 并添加 position embedding
        tok_emb = self.transformer.wte(idx) # (B, T, n_embd)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # (T, n_embd)
        # 广播机制
        x = tok_emb + pos_emb   # (B, T, n_embd)
        # 在 Transformer 块中前递
        for block in self.transformer.h:
            x = block(x)
        # 通过 GPT2 添加的归一化层
        x = self.transformer.ln_f(x)
        # 通过最后的线性层，转回词表
        x = self.lm_head(x) # (B, T, vocab_size)
        # 返回 logits ，训练时同时返回损失函数
        loss = None
        if labels is not None:
            # 交叉熵损失函数需要二维-一维输入，用于计算
            loss = F.cross_entropy(x.reshape(-1, x.size(-1)), labels.reshape(-1))
        return x, loss
        
    # 定义方法，导入 Hugging Face 下载的权重
    @classmethod
    def from_pretrained(cls, model_type):
        """Loads pretrained GPT-2 model weights from huggingface"""
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model
    
    # 添加优化器初始化：指定 weight decay 与使用 fused AdamW
    # 根据 GPT 论文，只对 2D 形状的参数做 weight decay
    def configure_optimizers(self, weight_decay, learning_rate, device):
        # 获取所有需要梯度的参数
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # 区分是否需要进行 weight dacay，只对 2D 形状的参数做
        # 如对所有参与矩阵乘法和嵌入层做权重衰退，而 bias 和层归一化的参数不做衰退
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        # 统计两种参数各自的个数并打印
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        # 只在主进程打印信息
        if master_process:
            print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
            print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # 使用 fused AdamW 即使用 cuda 内核融合后的优化器
        # 自动检测是否能使用 fused
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        if master_process:
            print(f"using fused AdamW: {use_fused}")
        # 指定优化器并返回
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
        return optimizer
    
# -----------------------------------------------------------------------------
import tiktoken
import numpy as np

# 从 np 文件中读取数据
def load_tokens(filename):
    npt = np.load(filename) # numpy token
    ptt = torch.tensor(npt, dtype=torch.long)   # pytorch token
    return ptt
    
# 训练数据生成 这里使用顺序固定点采样
# 修改为使用同目录文件夹下切片好的 FineWeb-Edu-10B Tokens 数据集
class DataLoaderLite:
    # 需要用于返回数据，能够处理 DDP 开启时的多卡不同数据生成
    def __init__(self, B, T, process_rank, num_processes, split):
        self.B = B
        self.T = T
        self.process_rank = process_rank    # 运行进程的编号，用于差异化返回数据
        self.num_processes = num_processes  # 总共有多少进程并行，用于每次步进
        
        # 读出磁盘数据到内存中
        assert split in {'train', 'val'}
        
        # 获取切片文件名
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)  # 获取路径下的所有文件名
        shards = [s for s in shards if split in s]  # 过滤出需要用的文件，区分训练集测试集
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]   # 组合出所有路径
        self.shards = shards    # 数据路径构成的 List
        
        assert len(shards) > 0, f"no shards found for split {split}"
        # 输出获取到的 split 类型（训练集/测试集）数据的切片文件总个数
        if master_process:
            print(f"found {len(shards)} shards for split {split}")
            
        # 初始化读取指针（状态）
        self.reset()
    
    def reset(self):    
        # 读取状态，记录当前读到哪个 batch
        self.cur_shard = 0  # 读到哪个切片
        self.tokens = load_tokens(self.shards[self.cur_shard])  # 当前切片的完整数据
        self.cur_pos = self.B * self.T * self.process_rank  # 读到当前切片的哪个 batch
        
    def next_batch(self):
        B, T = self.B, self.T
        # 将当前组需要的 token 读到 buf 中
        buf = self.tokens[self.cur_pos : self.cur_pos + B*T + 1]
        # 对 buf 错位切割得到一一对应的 x, y
        x = (buf[:-1]).reshape(B, T)
        y = (buf[1:]).reshape(B, T)
        # 更新当前位置
        self.cur_pos += B * T * self.num_processes
        # 如果下一个 Batch 对应的 buf 数据超过当前切片的 tokens 边界，重置并进入下一切片
        if self.cur_pos + B * T * self.num_processes + 1 > len(self.tokens):
            self.cur_shard = (self.cur_shard + 1) % len(self.shards)    # 循环使用数据
            self.tokens = load_tokens(self.shards[self.cur_shard])  # 加载新的切片
            self.cur_pos = self.B * self.T * self.process_rank  # 初始化位置
        return x, y

# -----------------------------------------------------------------------------
# 辅助函数，用于 hellaswag 评估
# 输入 tokens, mask, 和 logits, 返回模型输出的最小 loss 的标签，作为模型的输出
def get_most_likely_row(tokens, mask, logits):
    # 错位处理，使得模型输出的位置和真实数据对齐
    shift_logits = logits[:, :-1, :]
    shift_tokens = tokens[:, 1:]
    # 展平，计算交叉熵损失
    flat_shift_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.reshape(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.reshape(tokens.size(0), -1)
    # 使用掩码，计算关心位置（后续结尾）的平均损失和总损失
    shift_mask = mask[:, 1:]    # 掩码在 render_example 中只在后续结尾处取 1
    masked_shift_losses = shift_losses * shift_mask
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)
    # 根据计算结果，得到模型预测的结果，即取损失最小的后续结尾
    pred_norm = avg_loss.argmin().item()
    return pred_norm

# -----------------------------------------------------------------------------
# 数据并行初始化
"""
使用 torch 提供的多卡并行 DistributedDataParallel(DDP) 进行训练
运行无 DDP 版本（单卡）时：
python train_gpt2.py
运行 DDP 版本（多卡，以 8 卡为例）时：
torchrun --standalone --nproc_per_node=8 train_gpt2.py
AutoDL 运行前还需要：
unset OMP_NUM_THREADS
"""
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

# torchrun 命令会自动设置环境变量 RANK, LOCAL_RANK, 和 WORLD_SIZE
# 从这里开始，要想象有 ddp_world_size 个进程，同时运行整个代码
ddp = int(os.environ.get('RANK', -1)) != -1 # 检测是否使用了 torchrun
if ddp:
    # 使用 DDP 依赖于 cuda。这里需要设置这些设备，与 torchrun 生成的保持一致
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])  # 全局编号
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 本机编号
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # 设置主进程，用于打印日志，设置 checkpoint 等等
else:
    # 将 DDP 变量设置为单卡，用于给后续提供统一的输入参数格式
    # 单卡时相当于就是主进程
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # 自动检测设备
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print(f"using device: {device}")

# 设置初始化的随机种子
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
# 使用 torch 提供的神经网络专用编译器
use_compile = False # 是否使用的开关
if use_compile:
    model = torch.compile(model)
# 如果需要使用 ddp，需要把模型进行类型转换
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
# 保存一份未转换类型的模型，用于初始化优化器
raw_model = model.module if ddp else model

enc = tiktoken.get_encoding("gpt2")

# 使用梯度累加，实现和 GPT 论文中同样大小的 batch
# 先进行一些数值计算，得到需要多少组梯度进行累加
total_batch_size = 524288   # 2**19，524288，~0.5M，单位为 tokens。0.5M 和论文一致
B = 16   # 单个设备支持的 Batch_size，单位为 seqs
T = 1024    # 序列长度，GPT2 用 1024，GPT3 用 2048
assert total_batch_size % (B * T * ddp_world_size) == 0
grad_accum_steps = total_batch_size // (B * T * ddp_world_size)
if master_process:
    print(f"total desired batch size: {total_batch_size}")
    print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")

# 创建数据加载器实例
train_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="train")
val_loader = DataLoaderLite(B=B, T=T, process_rank=ddp_rank, num_processes=ddp_world_size, split="val")

# 在矩阵乘法运算中，使用 TF32(19bit) 代替 FP32(32bit)，以精读换速度和显存
torch.set_float32_matmul_precision('high')

# 学习率调度器（可变学习率函数），按照 GPT3 实现
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 715  # 约 375M Tokens
max_steps = 19073   # 刚好扫一遍数据集 10B Tokens
# 根据训练步数推进，改变学习率
def get_lr(it):
    # 线性 warmup
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps   # it+1 避免 0 学习率
    # 超过 max_steps 后使用 10% 最大学习率
    if it > max_steps:
        return min_lr
    # 中间段使用余弦函数衰减 先计算出系数
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    # 用系数进行余弦函数衰减
    return min_lr + coeff * (max_lr - min_lr)

# 使用优化器进行模型训练
# 按照 GPT3 论文设置超参数
optimizer = raw_model.configure_optimizers(weight_decay=0.1, learning_rate=6e-4, device=device)

# 在训练过程中写入日志，初始化日志
log_dir = "log"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"log.txt")
with open(log_file, "w") as f:
    pass

# GPT 原文实际训练超过 max_steps 次，但这里先用 max_steps。
# 这里的一个 step 在使用梯度累加后，达到和 GPT 论文一致，即 0.5M tokens
# 也就是这里是优化器的 step，设备计算的 step 被放进小循环中。
for step in range(max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1) # 记录最后一个 step 的标签
    
    # 每经过一段时间的训练，进行一次验证集评估
    if step % 250 == 0 or last_step:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_accum = 0.0    # 用于记录验证集总损失
            val_loss_steps = 20     # 验证集进行的 micro_step
            for _ in range(val_loss_steps):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(x, y)
                loss = loss / grad_accum_steps
                val_loss_accum += loss.detach()
        if ddp:
            dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
        if master_process:
            print(f"validation loss: {val_loss_accum.item():.4f}")
            # 将打印信息同步写入日志
            with open(log_file, "a") as f:
                f.write(f"{step} val {val_loss_accum.item():.4f}")
    
    # 每隔一段时间进行 Hellaswag 评估
    if (step % 250 == 0 or last_step) and (not use_compile):
        num_correct_norm = 0
        num_total = 0
        for i, example in enumerate(iterate_examples("val")):
            # 注意！需要考虑多进程的情况。
            # 每个进程只处理 hellaswag 评估中符合互斥条件的数据
            if i % ddp_world_size != ddp_rank:
                continue
            # 从 hellaswag 的函数读取张量验证数据
            _, tokens, mask, label = render_example(example)
            tokens = tokens.to(device)
            mask = mask.to(device)
            # 将数据进行前向传播，获取 logits
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(tokens)
                pred_norm = get_most_likely_row(tokens, mask, logits)
            num_total += 1
            num_correct_norm += int(pred_norm == label)
        # 处理完所有验证集数据后，多卡同步
        if ddp:
            # 在主进程初始化张量，用于 distributed 的同步操作
            num_total = torch.tensor(num_total, dtype=torch.long, device=device)
            num_correct_norm = torch.tensor(num_correct_norm, dtype=torch.long, device=device)
            # 对所有进程进行统一操作
            dist.all_reduce(num_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct_norm, op=dist.ReduceOp.SUM)
            # 转回 python 普通数据类型
            num_total = num_total.item()
            num_correct_norm = num_correct_norm.item()
        # 计算正确率
        acc_norm = num_correct_norm / num_total
        # 在主进程打印信息，并写入日志
        if master_process:
            print(f"HellaSwag accuracy: {num_correct_norm}/{num_total}={acc_norm:.4f}")
            with open(log_file, "a") as f:
                f.write(f"{step} hella {acc_norm:.4f}\n")
    
    # 添加评估的同时，运行模型进行输出。
    # 这部分代码不支持 torch.compile，会报错
    if ((step > 0 and step % 250 == 0) or last_step) and (not use_compile):
        model.eval()
        num_return_sequences = 4
        max_length = 32
        tokens = enc.encode("Hello, I'm a language model,")
        tokens = torch.tensor(tokens, dtype=torch.long)
        # 展开 Batch 维度，并进行重复指定次数
        tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
        xgen = tokens.to(device)
        # 这里使用独立的随机数生成器，让四个生成的句子不同
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(42 + ddp_rank)

        # 用循环实现逐步前推
        while xgen.size(1) < max_length:
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = model(xgen) # (B, T, vocab_size)
                # 取出最新的输出
                logits = logits[:, -1, :]   # (B, vocab_size)
                # 按照概率进行输出（随机取样）
                probs = F.softmax(logits, dim=-1)   # 获取模型输出的概率值
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)    # 取 topk 概率值
                ix = torch.multinomial(topk_probs, 1, generator=sample_rng)   # 按照概率，随机选择 topk 中的索引
                # 使用 gather 取出索引对应的 vocab_index
                xcol = torch.gather(topk_indices, -1, ix)
                # 将输出结果添加到 x 后方，用于继续向后预测
                xgen = torch.cat((xgen, xcol), dim=-1)

        # 打印模型输出的序列
        for i in range(num_return_sequences):
            tokens = xgen[i, :max_length].tolist() # 使用 tolist 将张量送回 CPU 并转回标准数据结构
            decoded = enc.decode(tokens)
            print(f"rank {ddp_rank} sample {i}: {decoded}")
    
    # 训练代码，进行一步优化
    model.train()
    optimizer.zero_grad()
    loss_accum = 0.0    # 统计总损失，用于打印信息
    # 用小循环实现梯度累加。每个 micro_step 是设备实际一次并行计算。
    for micro_step in range(grad_accum_steps):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        # 在计算过程中，进一步使用 BF16 来减少内存开销和数据传输开销
        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            logits, loss = model(x, y)
        # 注意！这里需要重新计算平均值，因为 torch 对每个 micro_step 的反向传播只做了累加。
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()
        # 使用 DDP 时，需要在最后一个 micro_step 进行多卡同步梯度
        # 所以必须在反向传播前声明需要同步
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    # loss_accum 只保存了本进程的值，因此也需要额外同步
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    # 添加梯度裁剪，控制模型优化的速度。函数返回裁剪前的梯度向量范数
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # 使用学习率调度器
    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr  # 这里实际上只有一个参数组，是 torch 要求用这种方式指定优化器学习率
    optimizer.step()
    
    torch.cuda.synchronize()    # 等待 GPU 计算完成
    t1 = time.time()
    dt = t1 - t0
    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps * ddp_world_size
    tokens_per_sec = tokens_processed / dt
    if master_process:    
        print(f"step {step:5d} | loss: {loss_accum.item():.6f} | lr: {lr:.4e} | norm: {norm:.4f} | dt: {dt*1000:.2f}ms | tok/sec: {tokens_per_sec:.2f}")
        with open(log_file, "a") as f:
            f.write(f"{step} train {loss_accum.item():.6f}\n")

# 释放进程
if ddp:
    destroy_process_group()
