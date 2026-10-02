"""
下载并准备 HellaSwag 给出的 LLM 评估数据集
https://github.com/rowanz/hellaswag

Example HellaSwag json item:

{"ind": 24, "activity_label": "Roof shingle removal", "ctx_a": "A man is sitting on a roof.", "ctx_b": "he", "ctx": "A man is sitting on a roof. he", "split": "val", "split_type": "indomain", "label": 3, "endings": ["is using wrap to wrap a pair of skis.", "is ripping level tiles off.", "is holding a rubik's cube.", "starts pulling up roofing on a roof."], "source_id": "activitynet~v_-JhWjGDPHMY"}

ind: dataset ID
activity_label: The ActivityNet or WikiHow label for this example
context: There are two formats. The full context is in ctx. When the context ends in an (incomplete) noun phrase, like for ActivityNet, this incomplete noun phrase is in ctx_b, and the context up until then is in ctx_a. This can be useful for models such as BERT that need the last sentence to be complete. However, it's never required. If ctx_b is nonempty, then ctx is the same thing as ctx_a, followed by a space, then ctx_b.
endings: a list of 4 endings. The correct index is given by label (0,1,2, or 3)
split: train, val, or test.
split_type: indomain if the activity label is seen during training, else zeroshot
source_id: Which video or WikiHow article this example came from

关于 HellaSwag 测试的详细信息，可以参考同名论文。

下面是 GPT 官方论文中给出的成绩：
gpt2 (124M)
- eleuther harness reports acc 28.92%, acc_norm 31.14% (multiple choice style)
- this script: 10042 acc: 0.2859 acc_norm: 0.2955 (completion style)

gpt2-xl (1558M)
- eleuther harness reports acc 40.04%, acc_norm 50.89% (multiple choice style)
- this script: 10042 acc: 0.3842 acc_norm: 0.4893 (completion style)

HellaSwag 数据集（用于验证、评估等等）总共有 10,042 个样本。
"""

import os
import json
import requests
import tiktoken
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import GPT2LMHeadModel

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "hellaswag")

# 从 url 下载数据到本地，命名为 fname，流式下载每次只加载 1024KB
def download_file(url: str, fname: str, chunk_size=1024):
    """辅助函数，用于从 url 下载文件"""
    resp = requests.get(url, stream=True)   # 从指定 url 流式读取数据
    total = int(resp.headers.get("content-length", 0))  # 从 HTTP 头获取总字节数
    with open(fname, "wb") as file, tqdm(
        desc=fname, # 进度条名称
        total=total,    # 总大小
        unit="iB",  # 单位，这里是字节
        unit_scale=True,    # 自动换算，例如 KB, MB, GB
        unit_divisor=1024,  # 使用 1024 进位制
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)

# 数据集下载地址
hellaswags = {
    "train": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_train.jsonl",
    "val": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl",
    "test": "https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_test.jsonl",
}

# GPT 编码器
enc = tiktoken.get_encoding("gpt2")

def download(split):
    """在 DATA_CACHE_DIR 处下载数据集"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)  # 创建目录
    data_url = hellaswags[split]
    data_filename = os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl")
    if not os.path.exists(data_filename):   # 防止重复下载
        print(f"Downloading {data_url} to {data_filename}...")
        download_file(data_url, data_filename)

# 读取数据集的迭代器
def iterate_examples(split):
    # 验证集共 10,042 个样本
    download(split)
    with open(os.path.join(DATA_CACHE_DIR, f"hellaswag_{split}.jsonl"), "r") as f:
        for line in f:
            example = json.loads(line)
            yield example

def render_example(example):
    """把一条 HellaSwag 样本（一个字典）转换成模型可以批量处理的张量"""
    ctx = example["ctx"]    # 样本的上下文
    label = example["label"]    # 样本的正确结尾
    endings = example["endings"]    # 样本的四个后续结尾
    
    # 初始化用于保存词元化后的数据
    data = {
    "label": label,
    "ctx_tokens": None,
    "ending_tokens": [],
    }
    
    # 遍历 4 个候选结尾，构建完整 token 序列和 mask
    ctx_tokens = enc.encode(ctx)
    data["ctx_tokens"] = ctx_tokens
    tok_rows = []
    mask_rows = []  # 掩蔽标签，用于遮蔽候选结尾信息，让模型去预测结尾
    for end in endings:
        end_tokens = enc.encode(" " + end)  # 在后续结尾前加空格，以符合英文句法
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0]*len(ctx_tokens) + [1]*len(end_tokens))
        data["ending_tokens"].append(end_tokens)
        
    # 转成 torch 张量
    # 因为不同的后续结尾长度可能不同，需要用赋值方式进行转换
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for i, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[i, :len(tok_row)] = torch.tensor(tok_row)
        mask[i, :len(mask_row)] = torch.tensor(mask_row)
    
    return data, tokens, mask, label

# 运行 OpenAI 提供权重的模型进行评估，用于程序上和数据上的参考
@torch.no_grad()
def evaluate(model_type, device):
    
    torch.set_float32_matmul_precision('high') # 使用 tf32
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    
    # 初始化统计量
    num_correct_norm = 0    # 基于平均损失判定预测正确的样本数
    num_correct = 0         # 基于总损失判定预测正确的样本数
    num_total = 0           # 已评估的样本总数
    
    for example in iterate_examples("val"):
        data, tokens, mask, label = render_example(example)
        tokens = tokens.to(device)
        mask = mask.to(device)
        
        # 获取模型输出
        logits = model(tokens).logits  # (4, max_len, vocab_size)
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
        pred = sum_loss.argmin().item()
        pred_norm = avg_loss.argmin().item()
        
        # 更新统计量
        num_total += 1
        num_correct += int(pred == label)
        num_correct_norm += int(pred_norm == label)
        print(f"{num_total} acc_norm: {num_correct_norm}/{num_total}={num_correct_norm/num_total:.4f}")
        
        # 打印调试信息
        if num_total < 10:
            print("---")
            print(f"Context:\n {example['ctx']}")
            print(f"Endings:")
            for i, end in enumerate(example["endings"]):
                print(f"{i} (loss: {avg_loss[i].item():.4f}) {end}")
            print(f"predicted: {pred_norm}, actual: {label}")

if __name__ == "__main__":
    # 命令行参数解析
    # 例如可以这样在终端中启动：python hellaswag.py -m gpt2 -d cuda
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", type=str, default="gpt2", help="the model type to use")
    parser.add_argument("-d", "--device", type=str, default="cuda", help="the device to use")
    args = parser.parse_args()
    
    evaluate(args.model_type, args.device)