import math
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F

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
        # 缩放点积计算注意力权重
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # 使用掩码 maseked_fill(条件，True时用以覆盖的值)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        # Softmax 后得到注意力权重
        att = F.softmax(att, dim=-1)
        # 计算加权平均值
        y = att @ v
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
    
# -----------------------------------------------------------------------------
model = GPT.from_pretrained('gpt2')
print("didn't crash yay!")