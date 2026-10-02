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

## 训练代码

```python
# 添加梯度裁剪，控制模型优化的速度。函数返回裁剪前的梯度向量范数
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

梯度裁剪，相当于控制优化器一次优化的步幅大小。在训练初期，通常能够避免大幅度优化，增强训练的稳定性。

需要注意的是，梯度裁剪相当于需要遍历一遍所有反向传播计算得到的梯度值，会带来性能下降。

---

```python
# 使用学习率调度器
lr = get_lr(step)
for param_group in optimizer.param_groups:
    param_group['lr'] = lr  # 这里实际上只有一个参数组，是 torch 要求用这种方式指定优化器学习率
```

实际上 torch 自带一些学习率调度器可以使用，位于 torch.optim.lr_scheduler 模块。这里因为 GPT 的学习率有 warmup，cos_decay 以及后续的平稳段，函数较复杂，我们使用自定义的方式。

使用自定义学习率函数时，torch 规定**必须用这种方法**给优化器传递学习率。

---

```python
# 先进行一些数值计算，得到需要多少组梯度进行累加
total_batch_size = 524288   # 2**19，~0.5M，单位为 tokens。0.5M 和论文一致
B = 2   # 单个设备支持的 Batch_size，单位为 seqs
T = 1024    # 序列长度，GPT2 用 1024，GPT3 用 2048
assert total_batch_size % (B * T) == 0
grad_accum_steps = total_batch_size // (B * T)

...

for step in range(max_steps):
    ...
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
        loss.backward()
    ...
```

梯度累加，**是在单个设备上，多次使用不同数据进行前向传播，多次计算反向传播后对梯度进行求和（对各个参数单独，torch 的 backward 默认逻辑就是累加）后，再进行参数更新（即优化器步）**。

也就是实际达到的效果等价于**使用了一个大批量**，使得每个 step 能够和论文中的大批量达到统一。即对于外层 step 看，使用了一个很大的批量；对于内层 micro_step，则是考虑单个设备的计算内存限制后，实际进行的多次并行计算。

这里注意一个细节，`loss = loss / grad_accum_steps`。torch 的损失函数默认 reduction=mean，即以平均值作为损失函数。这在梯度累加之前没有问题，但引入梯度累加之后，**不同 micro_step 虽然内部计算了平均值，但在累加时并没有计算平均而是直接相加**，造成总的损失函数比一个大 batch 的损失函数计算结果偏大。因此这里需要除 micro_steps 的数量，用以达到平均值。 

这一步过后整个模型与 GPT 基本一致，只是每一步的训练时间很长。因为是多个 micro_step 的累加，多个 **micro_step 因设备限制，相互是串行的**。只有 micro_step 内部是并行的。

---

下一步需要引入 DDP，即 DistributedDataParallel 分布式数据并行。最关键的是从多进程的角度考虑多卡数据并行。

实际上就是把一个单进程程序修改为一个多进程程序的过程（所有进程跑同一份程序）。除了添加 DDP 需要的代码之外，还需要注意
- 在终端中**使用 torchrun 命令**而不是 python 命令运行程序
- 如何实现**数据并行**（不同卡跑不同数据的 forward backward）
- 在何处进行**进程通信**（即多卡之间同步梯度信息和模型）
- 只在主进程**打印信息**（打印占用 GPU-CPU 通信，耗时）
- 运行完成后需要手动**回收进程**
多卡数据并行，**本质上还是为了构建一个大的 batch**，也就是把梯度累加的 micro_step 放到不同卡上并行，减少总运行时间。所以还需要调整 grad_accum_steps 计算相关的代码。

### DDP 相关的初始化

主要处理 torchrun 运行引入的环境变量，需要读取进系统内存中，用于后续操作设备。

```python
# torchrun 命令会自动设置环境变量 RANK, LOCAL_RANK, 和 WORLD_SIZE
# 从这里开始，要想象有 ddp_world_size 个进程，同时运行整个代码
ddp = int(os.environ.get('RANK', -1)) != -1 # 检测是否使用了 torchrun
if ddp:
    # 使用 DDP 依赖于 cuda。这里需要设置这些设备，与 torchrun 生成的保持一致
    assert torch.cuda.is_available(), "for now i think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])  # 全局的进程编号
    ddp_local_rank = int(os.environ['LOCAL_RANK'])  # 本主机中的进程编号
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # 设置主进程，用于打印日志，设置 checkpoint 等等

...
# 如果需要使用 ddp，需要把模型进行类型转换
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
# 保存一份未转换类型的模型，用于初始化优化器
raw_model = model.module if ddp else model
```

### 对 DataLoader 的修改

主要处理如何实现数据并行的问题。

```python
# 需要用于返回数据，能够处理 DDP 开启时的多卡不同数据生成
def __init__(self, B, T, process_rank, num_processes):
    ...
    # 记录当前读到哪个 batch
    self.cur_pos = self.B * self.T * self.process_rank

def next_batch(self):
    ...
    # 更新当前位置
    self.cur_pos += B * T * self.num_processes
    # 如果下一个 Batch 对应的 buf 数据超过 tokens 边界，重置
    if self.cur_pos + B * T * self.num_processes + 1 > len(self.tokens):
        self.cur_pos = self.B * self.T * self.process_rank
    return x, y
```

### 对训练代码的修改

主要处理在何处进行进程通信的问题。相关解释可以看 AIQA。

```python
for step in range(max_steps):
    ...
    for micro_step in range(grad_accum_steps):
        ...
        # 使用 DDP 时，需要在最后一个 micro_step 进行多卡同步梯度
        # 所以必须在反向传播前声明需要同步
        if ddp:
            model.require_backward_grad_sync = (micro_step == grad_accum_steps - 1)
        loss.backward()
    # loss_accum 只保存了本进程的值，因此也需要额外同步
    if ddp:
        dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG)
    ...
```

## Class GPT

```python
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
    ...
    # 使用 fused AdamW 即使用 cuda 内核融合后的优化器
    # 自动检测是否能使用 fused
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and 'cuda' in device
    print(f"using fused AdamW: {use_fused}")
    # 指定优化器并返回
    optimizer = torch.optim.AdamW(optim_groups, 
                                lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)
    return optimizer
```

因为需要较大幅度自定义优化器，这里用一个函数辅助进行。

实现两个主要功能：
- 对所有需要梯度的 2D 模型参数进行权重衰退
- 如果支持 cuda，使用内核融合后的优化器进行加速，即 fused AdamW

分别对应两个 torch 规定的用法：
- 在优化器初始化时，使用**字典列表**对模型参数进行分组指定。
- 使用 fused AdamW 的标准流程（先自动检测是否可用，再在优化器初始化时指定）

## fineweb.py

### 下载数据

```python
# 下载数据集
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")
```

用 huggingface 提供的 datasets 类型下载并加载数据。有关于这个函数的使用方式，以及返回类型的数据结构、内存特点，还有怎么加载本地下载好的有标准格式化的数据集，参考 AIQA 中的相关部分。

### 处理数据

```python
eot = enc._special_tokens['<|endoftext|>']  # 获得特殊词元索引
def tokenize(doc):
    # 词元化一个文档并返回 numpy 数组
    tokens = [eot]  # eot 实际上被设计为在每个文档的开头
    tokens.extend(enc.encode_ordinary(doc["text"]))
    ...
```

填充特殊词元，以及取出文本（即上面的 fw 迭代器指向的字典数据对象的 text 值）并做词元化。

---

**在 python 程序中，要实现真正的并行，必须用多进程**。

因为 Python 有 GIL，CPU 密集型任务用多进程才能真正并行。Global Interpreter Lock，全局解释器锁，作用是保证**同一时刻，一个 Python 进程里只有一个线程能执行 Python 字节码**。

```python
# 用 CPU 多进程处理所有文档并保存
nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    ...
    for tokens in pool.imap(tokenize, fw, chunksize=16):
```

imap 是多进程中的**调度器 + 数据收集器**。这里需要注意的是，和之前的 DDP 不同，主代码只在主进程中执行，进程池中的进程只运行调度器给的函数。

nprocs 指定需要使用的进程（CPU 核心）数量。with 在主进程中维护进程池，由 imap 分发任务给各个进程（函数，数据迭代器，一次发的数据数量），运行完后**按照顺序返回**，使得主进程中能够用 for 循环遍历各个进程的返回值。

对于其他任务，存在可能更加高效的无序返回的调度器 imap_unordered。本任务不能使用，否则文本序列的连续性将被破坏。

有关多进程编码逻辑的细节，参考 AIQA 的相关部分。Windows 直接这样使用进程池有 bug，参考 debug_log 相关部分。

## Class DataLoaderLite

将数据加载器读取数据的来源切换为 fineweb.py 处理得到的数据。需要新增对于切片的处理逻辑。

```python
class DataLoaderLite:
    # 需要用于返回数据，能够处理 DDP 开启时的多卡不同数据生成
    def __init__(self, B, T, process_rank, num_processes, split):
        ...
        # 读出磁盘数据到内存中
        assert split in {'train', 'val'}
        # 获取切片文件名
        data_root = "edu_fineweb10B"
        shards = os.listdir(data_root)  # 获取路径下的所有文件名
        shards = [s for s in shards if split in s]  # 过滤出需要用的文件，区分训练集测试集
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]   # 组合出所有路径
        self.shards = shards    # 数据路径构成的 List
        
        ...
        # 读取状态，记录当前读到哪个 batch
        self.cur_shard = 0  # 读到哪个切片
        self.tokens = load_tokens(self.shards[self.cur_shard])  # 当前切片的完整数据
        self.cur_pos = self.B * self.T * self.process_rank  # 读到当前切片的哪个 batch
        
    def next_batch(self):
        ...
        # 更新当前位置
        self.cur_pos += B * T * self.num_processes
        # 如果下一个 Batch 对应的 buf 数据超过当前切片的 tokens 边界，重置并进入下一切片
        if self.cur_pos + B * T * self.num_processes + 1 > len(self.tokens):
            self.cur_shard = (self.cur_shard + 1) % len(self.shards)    # 循环使用数据
            self.tokens = load_tokens(self.shards[self.cur_shard])  # 加载新的切片
            self.cur_pos = self.B * self.T * self.process_rank  # 初始化位置
        return x, y
```