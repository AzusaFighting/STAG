# 文本属性图的量化：语义与结构集成

> **官方实现** "Quantizing Text-attributed Graphs for Semantic-Structural Integration"，已被 **KDD'25** 接收。[[arXiv](https://arxiv.org/abs/2507.19526)]

**作者：** 薄建元¹、吴昊²、方圆¹  
¹新加坡管理大学，新加坡  
²北京师范大学  
📧 jybo.2020@smu.edu.sg, wuhao@bnu.edu.cn, yfang@smu.edu.sg

STAG 是一个自监督框架，通过量化方法将图表示学习与大型语言模型相结合。它无需来自源数据集或目标数据集的标注数据，即可实现真正的零样本学习。

## 主要特性

- 无需任何标注数据的自监督学习
- 用于有效结构-语义融合的软令牌分配策略
- 用于语义保留的分布对齐机制
- 灵活的推理策略，同时支持基于 LLM 和传统方法
- 跨不同领域的真正零样本学习能力

## 文件结构

```bash
STAG/
├── configs/                    # 配置文件
│   ├── csv/                    # CSV 格式的结果
│   ├── log/                    # 训练日志
│   └── *.yaml                  # 不同实验的配置
├── model/                      # 模型实现
│   ├── __init__.py             # 模型构建器
│   ├── edcoder.py              # 编码器-解码器架构
│   └── fusion.py               # 特征融合模块
├── src/                        # 源代码
│   ├── config.py               # 配置工具
│   ├── gen_data.py             # 数据生成
│   └── lc_sampler.py           # 图采样
├── checkpoint/                 # 模型检查点
│   └── *_checkpoint.pt
├── codebook/                   # 码本
│   ├── subword_embeddings.pth
│   ├── subword_vocabulary.npy
├── dataset/                    # 原始图数据集
│   ├── cora/
│   │   ├── cora_graph.pth
│   │   ├── cora_metadata.pth
│   │   └── cora_text.pkl
│   ├── citeseer/
│   │   ├── citeseer_graph.pth
│   │   ├── cora_metadata.pth
│   │   └── cora_text.pkl
│   └── cora_full/
│       ├── cora_full_graph.pth
│       ├── cora_full_metadata.pth
│       └── cora_full_text.txt
└── lc_ego_graphs/             # 预计算的自我图样本
    ├── cora-lc-ego-graphs-64.pt
    ├── citeseer-lc-ego-graphs-64.pt
    ├── pubmed-lc-ego-graphs-64.pt
    └── ogbn-products-lc-ego-graphs-64.pt
```

## 大型文件（可在共享文件夹中获取）

以下大型文件/目录未包含在此仓库中，但可在共享文件夹中获取：

- `codebook/`：预训练的图分词器码本
- `dataset/`：处理后的图数据集（Cora、CiteSeer 等）
- `lc_ego_graphs/`：预计算的自我图样本

## 安装与配置

1. 从 [Google Drive](https://drive.google.com/drive/folders/1VoL3IbYSjJKF3JoUaJw6FZ4FBCrAHLlK?usp=drive_link) 下载所需文件
2. 创建并激活 conda 环境

```bash
conda env create -f environment.yml
conda activate stag
```

## 大型语言模型（LLM）配置

要使用 LLM，您需要在 [Hugging Face](https://huggingface.co/) 上创建账户，并申请访问特定模型，例如 [LLaMA-2](https://huggingface.co/meta-llama/Llama-2-7b) 和 [LLaMA-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B)。

## 运行实验

### 1. 预训练

预训练 STAG 模型：

```bash
# 在 Cora Full 数据集上预训练
python train.py --config configs/cora_full_pretrain.yaml
```

### 保存的模型检查点

模型检查点将保存在 `checkpoint/` 目录中。要在测试时使用特定检查点，请在配置文件中指定检查点路径：

```yaml
# 示例 config.yaml
checkpoint_path: "checkpoint/<MODEL_CHECKPOINT>.pt"  # 替换为您的检查点文件名
```

### 2. 测试小样本学习

#### 基于 LLM 的小样本学习

```bash
# 在 Cora 上测试 5-way 5-shot 学习
python test_few_shot_llm.py --config configs/cora-5-way-5-shot-llm.yaml
```

#### 不使用 LLM 的小样本学习（线性探针）

```bash
# 在 Cora 上测试 5-way 5-shot 学习
python test_few_shot_linear_probing.py --config configs/cora-5-way-5-shot-lb.yaml
```

#### 基于 LLM 的提示调优

对于不同的数据集，您需要在配置文件中调整提示调优的超参数：

- `batch_size_f`：提示调优的批次大小
- `lr_f`：提示调优的学习率
- `weight_decay_f`：提示调优的权重衰减
- `max_epoch_f`：提示调优的训练轮数
- `tau_f`：提示调优的温度参数

```bash
# 在 Cora 上使用提示调优测试 5-way 5-shot 学习
python test_few_shot_prompt_tuning_llm.py --config configs/cora-5-way-5-shot-pt-llm.yaml
```

#### 不使用 LLM 的提示调优

```bash
# 在 Cora 上使用提示调优测试 5-way 5-shot 学习
python test_few_shot_prompt_tuning.py --config configs/cora-5-way-5-shot-pt.yaml
```

### 3. 零样本学习

#### 使用 LLM 的零样本学习

```bash
# 在 Cora 上使用 LLM 测试 5-way 0-shot 学习
python test_zero_shot_llm.py --config configs/cora-5-way-0-shot-llm.yaml
```

#### 不使用 LLM 的零样本学习

```bash
# 在 Cora 上使用类别专属码本测试 5-way 0-shot 学习
python test_zero_shot.py --config configs/cora-5-way-0-shot.yaml
```
