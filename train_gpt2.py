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
    
# -----------------------------------------------------------------------------
# 自动检测设备
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
print(f"using device: {device}")

# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.to(device)

# 使用分词器（需要与 GPT2 一致）生成示例样本
import tiktoken
# 从 input.txt 中读取文本数据
with open('input.txt', 'r') as f:
    text = f.read()
text = text[:1000]
# 获取 GPT2 的分词器，用于生成 vocab_index 向量
enc = tiktoken.get_encoding('gpt2')
tokens = enc.encode(text)
# 整理成用于模型训练的 Batch
B, T = 4, 32
# +1 为了让最后一个 seq 的末尾位置有对应的 y
buf = torch.tensor(tokens[:B*T + 1])
# 错位截取 buf，得到 x 一一对应 y
x = buf[:-1].reshape(B, T).to(device)
y = buf[1:].reshape(B, T).to(device)

# 检查目前的 Batch 操作是否能够正常运行
logits, loss = model(x, y)
print(loss)
import sys; sys.exit(0)

# 使用训练好的模型进行预测
model.eval()
num_return_sequences = 5
max_length = 30
tokens = enc.encode("Hello, I'm a language model,")
tokens = torch.tensor(tokens, dtype=torch.long)
# 展开 Batch 维度，并进行重复指定次数
tokens = tokens.unsqueeze(0).repeat(num_return_sequences, 1)
x = tokens.to(device)

# 使用 GPT 进行推理，x (B, T)
# 统一随机种子便于复现
torch.manual_seed(42)
torch.cuda.manual_seed(42)
# 用循环实现逐步前推
while x.size(1) < max_length:
    with torch.no_grad():
        logits = model(x)
        # 取出最新的输出
        logits = logits[:, -1, :]   # (B, vocab_size)
        # 按照概率进行输出（随机取样）
        probs = F.softmax(logits, dim=-1)   # 获取模型输出的概率值
        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)    # 取 topk 概率值
        ix = torch.multinomial(topk_probs, 1)   # 按照概率，随机选择 topk 中的索引
        # 使用 gather 取出索引对应的 vocab_index
        xcol = torch.gather(topk_indices, -1, ix)
        # 将输出结果添加到 x 后方，用于继续向后预测
        x = torch.cat((x, xcol), dim=-1)

# 打印模型输出的序列
for i in range(num_return_sequences):
    tokens = x[i, :max_length].tolist() # 使用 tolist 将张量送回 CPU 并转回标准数据结构
    decoded = enc.decode(tokens)
    print(">", decoded)