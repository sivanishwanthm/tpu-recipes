# Reproducible Benchmark Recipes for TPUs

This repository contains production-ready instructions and configurations to reproduce training and serving benchmarks for large machine learning models using Google Cloud TPUs. The focus is on reliably achieving performance metrics (like throughput/TFLOPS) across various hardware and software stacks.

## How to Use

1. **Identify your requirements**: Determine the model, TPU type, framework, and workload you are interested in.
2. **Select a recipe**: Use the Benchmark Support Matrix table below to find a recipe that meets your needs.
3. **Follow the recipe**: Each recipe link provides procedures to prepare your environment, run the benchmark, and analyze the results.

## Benchmark Support Matrix

### TPU Training Benchmarks

| Models | TPU Machine Type | Framework / Library | Workload Type | Orchestrator | Link to the recipe |
| :--- | :--- | :--- | :--- | :--- | :--- |
| GPT-3 175B | Trillium (v5p) | MaxText | Pre-training | GKE / XPK | [Link](./training/trillium/GPT3-175B-MaxText) |
| Gemma-3 12B | Trillium (v5p) | MaxText | Pre-training | GKE / XPK | [Link](./training/trillium/Gemma3-12B-MaxText) |
| Llama-2 70B | Trillium (v5p) | MaxText | Pre-training | GKE / XPK | [Link](./training/trillium/Llama2-70B-MaxText) |
| Llama-3.1 405B | Trillium (v5p) | MaxText | Pre-training | GKE / XPK | [Link](./training/trillium/Llama3.1-405B-MaxText) |
| Llama-3.1 70B | Trillium (v5p) | MaxText | Pre-training | GKE / XPK | [Link](./training/trillium/Llama3.1-70B-MaxText) |
| Llama-3.1 8B | Trillium (v5p) | MaxText | Pre-training | GKE / XPK | [Link](./training/trillium/Llama3.1-8B-MaxText) |
| Mistral 7B | Trillium (v5p) | MaxText | Pre-training | GKE / XPK | [Link](./training/trillium/Mistral-7B-MaxText) |
| Mixtral 8x22B | Trillium (v5p) | MaxText | Pre-training | GKE / XPK | [Link](./training/trillium/Mixtral-8x22B-MaxText) |
| Mixtral 8x7B | Trillium (v5p) | MaxText | Pre-training | GKE / XPK | [Link](./training/trillium/Mixtral-8x7B-MaxText) |
| DLRM-V2 | v5p | Tensorflow | Training | GKE / XPK | [Link](./training/v5p/DLRM-V2-Tensorflow) |
| DeepSeek-3 671B | v5p | MaxText | Pre-training | GKE / XPK | [Link](./training/v5p/DeepSeek3-671B-MaxText) |
| Diffusion 2 | v5p | MaxDiffusion | Training | GKE / XPK | [Link](./training/v5p/Diffusion-2-MaxDiffusion) |
| GPT-3 175B | v5p | MaxText | Pre-training | GKE / XPK | [Link](./training/v5p/GPT3-175B-MaxText) |
| Llama-2 7B | v5p | MaxText | Pre-training | GKE / XPK | [Link](./training/v5p/Llama2-7B-MaxText) |
| Llama-3.1 405B | v5p | MaxText | Pre-training | GKE / XPK | [Link](./training/v5p/Llama3.1-405B-MaxText) |
| Llama-4 Maverick 17B 128E | v5p | MaxText | Pre-training | GKE / XPK | [Link](./training/v5p/Llama4-Maverick-17B-128E-MaxText) |
| Llama-4 Scout 17B 16E | v5p | MaxText | Pre-training | GKE / XPK | [Link](./training/v5p/Llama4-Scout-17B-16E-MaxText) |
| Mixtral 8x7B | v5p | MaxText | Pre-training | GKE / XPK | [Link](./training/v5p/Mixtral-8x7B-MaxText) |
| SDXL | v5p | MaxDiffusion | Training | GKE / XPK | [Link](./training/v5p/SDXL-MaxDiffusion) |
| DeepSeek-3 671B | Ironwood | MaxText | Pre-training | GKE / XPK | [Link](./training/ironwood/deepseek3-671b) |
| GPT-OSS 120B | Ironwood | MaxText | Pre-training | GKE / XPK | [Link](./training/ironwood/gpt-oss-120b) |
| Llama-3.1 405B | Ironwood | MaxText | Pre-training | GKE / XPK | [Link](./training/ironwood/llama3.1-405b) |
| Llama-3.1 70B | Ironwood | MaxText | Pre-training | GKE / XPK | [Link](./training/ironwood/llama3.1-70b) |
| Qwen-3 235B A22B | Ironwood | MaxText | Pre-training | GKE / XPK | [Link](./training/ironwood/qwen3-235b-a22b) |

### TPU Inference Benchmarks

| Models | TPU Machine Type | Framework / Library | Workload Type | Orchestrator | Link to the recipe |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Llama, Gemma, Mixtral | Trillium (v6e) | MaxText | Inference | JetStream | [Link](./inference/trillium/JetStream-MaxText) |
| Llama, Gemma, Mixtral | Trillium (v6e) | PyTorch | Inference | JetStream | [Link](./inference/trillium/JetStream-Pytorch) |
| SDXL, PaliGemma | Trillium (v6e) | MaxDiffusion | Inference | - | [Link](./inference/trillium/MaxDiffusion) |
| Llama, Gemma, Mixtral | Trillium (v6e) | vLLM | Inference | - | [Link](./inference/trillium/vLLM) |
| Llama, Gemma, Mixtral | v5e | MaxText | Inference | JetStream | [Link](./inference/v5e/JetStream-MaxText) |
| Llama, Gemma, Mixtral | v5e | PyTorch | Inference | JetStream | [Link](./inference/v5e/JetStream-Pytorch) |
| SDXL, PaliGemma | v5e | MaxDiffusion | Inference | - | [Link](./inference/v5e/MaxDiffusion) |

### TPU Microbenchmarks

| Benchmark | TPU Machine Type | Framework / Library | Workload Type | Orchestrator | Link to the recipe |
| :--- | :--- | :--- | :--- | :--- | :--- |
| HBM Bandwidth | All | Python | Microbenchmark | - | [Link](./microbenchmarks/benchmark_hbm.py) |
| Matmul | All | Python | Microbenchmark | - | [Link](./microbenchmarks/benchmark_matmul.py) |
| Collectives | Trillium | - | Microbenchmark | - | [Link](./microbenchmarks/trillium/collectives) |

## Repository Organization

- **./training**: Instructions to reproduce training performance for various models (LLMs, Diffusion).
- **./inference**: Instructions to reproduce inference performance.
- **./microbenchmarks**: Low-level benchmarks (matrix multiplication, memory bandwidth).

## Contributor notes

Note: This is not an officially supported Google product. This project is not
eligible for the [Google Open Source Software Vulnerability Rewards
Program](https://bughunters.google.com/open-source-security).
