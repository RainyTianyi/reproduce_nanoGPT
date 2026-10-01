"""
加载 FineWeb-Edu 10B子集 数据集用于模型预训练。
https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
下载并词元化文本数据，同时以 100M tokens 为一个切片保存到磁盘中。
运行方式：
$ python fineweb.py
将保存到运行路径下的 "edu_fineweb10B" 文件夹中。
"""

import os

# 可选，需要注册账号并获取只读 token
# os.environ["HF_TOKEN"] = "hf_mytoken"
# 指定国内镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# 缓存重定向为文件存储区，不占用系统盘和数据盘
os.environ["HF_HOME"] = "/root/autodl-fs/hf_cache_fineweb/"

import multiprocessing as mp
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

# 设置基本信息
local_dir = "edu_fineweb10B"
remote_name = "sample-10BT"
shard_size = int(1e8)   # 每个切片 100M tokens

# 创建路径
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(DATA_CACHE_DIR, exist_ok=True)

# 下载数据集
fw = load_dataset("HuggingFaceFW/fineweb-edu", name=remote_name, split="train")

# 初始化 GPT2 tokenizer
enc = tiktoken.get_encoding("gpt2")
eot = enc._special_tokens['<|endoftext|>']  # 获得特殊词元索引
def tokenize(doc):
    # 词元化一个文档并返回 numpy 数组
    tokens = [eot]  # eot 实际上被设计为在每个文档的开头
    tokens.extend(enc.encode_ordinary(doc["text"]))
    tokens_np = np.array(tokens)
    # 节省空间，切换为 uint16 存储
    assert (0 <= tokens_np).all() and (tokens_np < 2**16).all(), "token dictionary too large for uint16"
    tokens_np_uint16 = tokens_np.astype(np.uint16)
    return tokens_np_uint16

# numpy 文件写入辅助函数
def write_datafile(filename, tokens_np):
    np.save(filename, tokens_np)

# 用 CPU 多进程处理所有文档并保存
nprocs = max(1, os.cpu_count() // 2)
with mp.Pool(nprocs) as pool:
    shard_index = 0 # 初始化切片编号
    all_tokens_np = np.empty((shard_size,), dtype=np.uint16)    # 初始化存储数组
    token_count = 0
    progress_bar = None # 进度条
    for tokens in pool.imap(tokenize, fw, chunksize=16):
        
        # 判断当前切片是否有足够空间放下新的词元化后的文档
        if token_count + len(tokens) < shard_size:
            # 空间够，append 到存储数组后
            all_tokens_np[token_count: token_count + len(tokens)] = tokens
            token_count += len(tokens)
            # 更新进度条，使用 tqdm 进度条库
            if progress_bar is None:
                progress_bar = tqdm(total=shard_size, unit="tokens", desc=f"Shard {shard_index}")
            progress_bar.update(len(tokens))
        else:
            # 当前切片空间不足，创建新的切片空间，将文档拆分写入
            # 将第 0 个切片作为验证集
            split = 'val' if shard_index == 0 else "train"
            filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
            # 拆分文档，填满当前切片
            remainder = shard_size - token_count
            progress_bar.update(remainder)
            all_tokens_np[token_count: token_count + remainder] = tokens[:remainder]
            write_datafile(filename, all_tokens_np)
            # 创建新切片空间，这里用覆盖写入节省内存
            shard_index += 1
            progress_bar = None
            all_tokens_np[0: len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder
        
        # 处理最后一个切片的剩余文档
    if token_count != 0:
        split = "val" if shard_index == 0 else "train"
        filename = os.path.join(DATA_CACHE_DIR, f"edufineweb_{split}_{shard_index:06d}")
        write_datafile(filename, all_tokens_np[:token_count])