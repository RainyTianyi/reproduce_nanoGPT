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

--------------------

```python
# 按照概率进行输出（随机取样）
probs = F.softmax(logits, dim=-1)   # 获取模型输出的概率值
topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)    # 取 topk 概率值
ix = torch.multinomial(topk_probs, 1)   # 按照概率，随机选择 topk 中的索引
# 使用 gather 取出索引对应的 vocab_index
xcol = torch.gather(topk_indices, -1, ix)
```

只保留概率最高的 k 个 token，把其余低概率 token 全部排除，然后只在这 k 个候选里按概率采样。
topk_probs, topk_indices 的形状都是 (B, 50)。

multinomial(prob_list, n_sample) 按 prob_list 相对权重采样 n_sample 次。这里返回采样结果对应的 prob_list 索引 (B, 1)。

torch.gather(input, dim, index)沿指定维度，根据索引张量 index 从输入张量 input 中逐个元素地“收集”值。这里返回 vocab 索引 (B, 1)。

--------------------

```python
# GPT 中使用了 embedding 权重共享
# 即 token embedding 和最后的线性层共享权重，减少大量参数的同时提高模型性能
self.transformer.wte.weight = self.lm_head.weight
```

线性层默认存储的矩阵形状顺序和创建时是相反的，即使用 nn.Linear(n_i, n_o) 得到的权重矩阵形状为 (n_o, n_i)，这里恰好与 token embedding 层形状一致。

同时利用 nn.Module 对赋值操作的重写，实现了这两层的 weight 指向显存中同一块矩阵（nn.Parameter）。

--------------------

# SECTION2

## 训练代码

```python
# 在矩阵乘法运算中，使用 TF32(19bit) 代替 FP32(32bit)，以精读换速度和显存
torch.set_float32_matmul_precision('high')
```

下面是 Nvidia 文档中关于 TF32 的介绍。
![TF32](<pic/TF32 in GEMM.png>)

在矩阵乘法运算中，使用 TF32(19bit) 代替 FP32(32bit)，理想情况下可以带来 8 倍的速度提升。

`torch.set_float32_matmul_precision` 控制 **float32 矩阵乘法（matmul）的内部计算精度**，在**性能**和**数值精度**之间进行权衡：允许内部计算使用较低精度的数据类型（如 TF32 或 bfloat16），显著提升矩阵运算速度，输出的数据类型仍然是 float32。

### 三种精度模式

该函数接受一个字符串参数，有三种可选值，其内部计算机制和精度/性能对比如下：

| 模式 | 内部计算数据类型 | 精度（尾数位） | 相对性能 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **`"highest"`** (默认) | **float32** | 23 位显式存储 | 基准（较低） | 使用完整的 float32 精度进行计算，数值最精确，但速度最慢。 |
| **`"high"`** | **TensorFloat32 (TF32)** 或 **bfloat16_3x** | 约 10-14 位 | 较高 | 优先使用 TF32（10 位尾数）；若无支持，则采用一种基于 3 个 bfloat16 数的算法（约 14 位有效尾数），在精度和速度间取得良好平衡。 |
| **`"medium"`** | **bfloat16** | 7 位显式存储 | 最高 | 直接使用 bfloat16 进行内部计算，速度最快，但精度损失最大。若硬件不支持快速的 bfloat16 矩阵乘法，则会回退到 `"high"` 模式。 |

> **关于 `"high"` 模式的 bfloat16_3x 算法**：其原理是将一个 float32 数拆分为三个 bfloat16 数的和（因为 float32 的 23 位尾数 ≈ 3 × 7 位 bfloat16 尾数）。两个 float32 相乘可表示为 9 个 bfloat16 乘积之和，`"high"` 模式仅保留其中最重要的 3 个乘积，从而在利用 bfloat16 高速运算单元的同时，尽可能保留精度。

### 行为与注意事项

*   **仅影响 CUDA 设备**
*   **不改变输出 dtype**
*   **不影响卷积运算**：卷积操作的精度由其他标志控制，例如 `torch.backends.cudnn.allow_tf32`。
*   **硬件依赖**：`"high"` 和 `"medium"` 模式带来的加速主要依赖于 **NVIDIA Ampere 架构（如 A100、RTX 30 系列）及更新的 GPU**，因为它们支持 TF32 和 bfloat16 的 Tensor Core 加速。在 Volta (V100) 等较旧架构上，设置这些模式可能不会带来性能提升。
*   **与 `allow_tf32` 的等价关系**：
    *   设置为 `"highest"` 等价于 `torch.backends.cuda.matmul.allow_tf32 = False`。
    *   设置为 `"high"` 或 `"medium"` 等价于 `torch.backends.cuda.matmul.allow_tf32 = True`。

---

```python
# 在计算过程中，进一步使用 BF16 来减少内存开销和数据传输开销
with torch.autocast(device_type=device, dtype=torch.bfloat16):
    logits, loss = model(x, y)
```

BF16 比传统的 FP16 用精度换取更大（达到 FP32 和 TF32）的数值表示范围，避免了在网络训练过程中的 gradient scaling 操作。但我们不希望在所有的网络值上都降低精读，例如损失函数的计算。框架帮我们实现了这一点，即只对部分运算采用 BF16 而对精度敏感的计算仍使用 FP32。具体可以查框架的文档。

需要注意的是，pytorch autocast 规定**不要在模型中显式使用**如 dtype=torch.bfloat16 这样的声明，而是使用 with 块括起模型的前向传播部分。同时，**不建议把反向传播和优化器计算**放到 autocast 块中。

---

```python
# 使用 torch 提供的神经网络专用编译器
model = torch.compile(model)
```

torch 为自己的 model 提供了专用的编译器，**用编译时间的较少增长换取网络训练时间的较大减少**。大部分情况下，我们希望默认使用 torch.compile，除了极少数的 debug 情况。

python 编译器在编译 model 计算（前向传播和反向传播）时，会严格按照编码步骤一步一步实现，这样得到的程序在 GPU 上运行时，无法知道下一步要做什么数值运算。这样就需要 GPU 反复读写显存带来大量开销。

而 torch 编译器**会读完整个模型代码**，然后把整个模型当作一个整体做实现。相似的步骤（如 GELU 中大量对矩阵逐个值操作）会一次性在 GPU 内运算完成，大大减少读写开销。（这里说的本质上是一种内核融合技术，是 torch.compile 用于加速的方法之一）

值得注意的是**存在 torch.compile 无法加速的操作**，如下面要引入的 Flash Attention。

---

```python
model = GPT(GPTConfig(vocab_size=50304))
```

50257 对于 GPU 来说不是一个好的数字，因为硬件并行基本上是以 2 的幂次设计的。对于非 2 的幂次的数据，GPU 只能先处理满足自己结构的数据，完成后再单独处理剩下的数据，这部分单独处理开销不小。

因此我们考虑**使用 pad 方法将模型中所有数据个数尽可能贴近 2 的幂次**，例如这里的 50304=128*393。

## CausalSelfAttention

```python
# 缩放点积计算注意力权重
att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
# 使用掩码 maseked_fill(条件，True时用以覆盖的值)
att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
# Softmax 后得到注意力权重
att = F.softmax(att, dim=-1)
# 计算加权平均值
y = att @ v
```

针对这四步运算，因为涉及到注意力操作的运算设计，torch.compile 是无法识别这里的内核融合操作的，也就是会保留逐步运算的实现（约 17ms）。Flash Attention 通过 online softmax 技巧**将注意力操作的运算进行内核融合**，使得一次运算的时间减少至约 3ms。（可以参考 Flash Attention 论文以及 online softmax 论文）

```python
# 使用 torch 实现的 Flash Attention
y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

实际上 Flash Attention 比传统的注意力计算有更多的浮点数运算次数，但由于其极大减少了 GPU 访存次数，带来的加速效果可以达到约 7 倍。这就指出了在优化计算时间时，**优化访存方式往往比优化计算速度更加有效**。

# SECTION3

