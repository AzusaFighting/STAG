# Quantizing Text-attributed Graphs for Semantic-Structural Integration

[中文文档](README_zh.md)

> **Official Implementation** of "Quantizing Text-attributed Graphs for Semantic-Structural Integration" accepted at **KDD'25**. [[arXiv](https://arxiv.org/abs/2507.19526)]

**Authors:** Jianyuan Bo¹, Hao Wu², Yuan Fang¹  
¹Singapore Management University, Singapore  
²Beijing Normal University  
📧 jybo.2020@smu.edu.sg, wuhao@bnu.edu.cn, yfang@smu.edu.sg

STAG is a self-supervised framework that bridges graph representation learning and large language models through a quantization approach. It enables true zero-shot learning without requiring labeled data from either source or target datasets.

## Key Features

- Self-supervised learning without requiring any labeled data
- Soft token assignment strategy for effective structural-semantic integration
- Distribution alignment mechanism for semantic preservation
- Flexible inference strategies supporting both LLM-based and traditional approaches
- True zero-shot learning capabilities across different domains

## File Structure

```bash
STAG/
├── configs/                    # Configuration files
│   ├── csv/                    # Results in CSV format
│   ├── log/                    # Training logs
│   └── *.yaml                  # Configs for different experiments
├── model/                      # Model implementations
│   ├── __init__.py             # Model builder
│   ├── edcoder.py              # Encoder-decoder architecture
│   └── fusion.py               # Feature fusion modules
├── src/                        # Source code
│   ├── config.py               # Configuration utilities
│   ├── gen_data.py             # Data generation
│   └── lc_sampler.py           # Grah sampling
├── checkpoint/                 # Model checkpoints
│   └── *_checkpoint.pt
├── codebook/                   # codebooks
│   ├── subword_embeddings.pth
│   ├── subword_vocabulary.npy
├── dataset/                    # Raw graph datasets
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
└── lc_ego_graphs/             # Pre-computed ego-graph samples
    ├── cora-lc-ego-graphs-64.pt
    ├── citeseer-lc-ego-graphs-64.pt
    ├── pubmed-lc-ego-graphs-64.pt
    └── ogbn-products-lc-ego-graphs-64.pt
```

## Large Files (Available in Share Folder)

The following large files/directories are not included in this repository but are available in the share folder:

- `codebook/`: Pre-trained graph tokenizer codebooks
- `dataset/`: Processed graph datasets (Cora, CiteSeer, etc.)
- `lc_ego_graphs/`: Pre-computed ego-graph samples

## Setup and Installation

1. Download required files from [Google Drive](https://drive.google.com/drive/folders/1VoL3IbYSjJKF3JoUaJw6FZ4FBCrAHLlK?usp=drive_link)
2. Create and activate conda environment

```bash
conda env create -f environment.yml
conda activate stag
```

## LLM Setup

​To use the LLM, you'll need to create an account on [Hugging Face](https://huggingface.co/) and apply for access to specific models such as [LLaMA-2](https://huggingface.co/meta-llama/Llama-2-7b) and [LLaMA-3](https://huggingface.co/meta-llama/Meta-Llama-3-8B).

## Running Experiments

### 1. Pretraining

To pretrain the STAG model:

```bash
# pretrain on Cora Full
python train.py --config configs/cora_full_pretrain.yaml
```

### Saved Model Checkpoints

Model checkpoints will be saved in the `checkpoint/` directory. To use a specific checkpoint for testing, specify the checkpoint path in your config file:

```yaml
# Example config.yaml
checkpoint_path: "checkpoint/<MODEL_CHECKPOINT>.pt"  # Replace with your checkpoint filename
```

### 2. Test Few-shot Learning

#### Few-shot Learning with LLM

```bash
# Test 5-way 5-shot learning on Cora
python test_few_shot_llm.py --config configs/cora-5-way-5-shot-llm.yaml
```

#### Few-shot Learning without LLM (Linear Probing)

```bash
# Test 5-way 5-shot learning on Cora
python test_few_shot_linear_probing.py --config configs/cora-5-way-5-shot-lb.yaml
```

#### Prompt Tuning with LLM

For different datasets, you'll need to tune the prompt tuning hyperparameters in the config file:

- `batch_size_f`: Batch size for prompt tuning
- `lr_f`: Learning rate for prompt tuning
- `weight_decay_f`: Weight decay for prompt tuning
- `max_epoch_f`: Number of epochs for prompt tuning
- `tau_f`: Temperature parameter for prompt tuning

```bash
# Test 5-way 5-shot learning on Cora with prompt tuning
python test_few_shot_prompt_tuning_llm.py --config configs/cora-5-way-5-shot-pt-llm.yaml
```

#### Prompt Tuning without LLM

```bash
# Test 5-way 5-shot learning on Cora with prompt tuning
python test_few_shot_prompt_tuning.py --config configs/cora-5-way-5-shot-pt.yaml
```

### 3. Zero-shot Learning

#### Zero-shot with LLM

```bash
# Test 5-way 0-shot learning on Cora with LLM
python test_zero_shot_llm.py --config configs/cora-5-way-0-shot-llm.yaml
```

#### Zero-shot without LLM

```bash
# Test 5-way 0-shot learning on Cora with class-specific codebook
python test_zero_shot.py --config configs/cora-5-way-0-shot.yaml
```
