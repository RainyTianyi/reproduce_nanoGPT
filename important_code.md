# SECTION1

## CausalSelfAttention

```python
self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
```

使用 register_buffer 注册**不参与梯度更新（不可学习）**，但又是模型状态一部分的张量。

使用 tril 对矩阵进行对角线切割（保留对角线及下三角，用于生成掩码）。

使用 reshape 增加批量维度和序列长度维度，便于后续使用广播机制。



## Class GPT

```python
# 定义核心模型，使用 ModuleDict 自定义层的名称
self.transformer = nn.ModuleDict(dict(
    wte = nn.Embedding(config.vocab_size, config.n_embd),   # token embedding
    wpe = nn.Embedding(config.block_size, config.n_embd),   # position embedding
    h = nn.ModuleList([Block(config) for _ in range(config.n_layer)])   # Transformer 解码器层
    ln_f = nn.LayerNorm(config.n_embd) # GPT2 新增层归一化
))
```

使用 ModuleDict 自定义 torch 中网络层的名称。这里是为了把所有的中间层放进名为 transformer 的区域中，使得某一层的名字形如：

```
transformer.wte.weight
transformer.h.0.attn.c_attn.weight
transformer.h.0.attn.c_attn.bias
```

