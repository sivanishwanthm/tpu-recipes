# Cloud TPU performance benchmark recipes

This repository contains recipes that provide instructions to reproduce specific
workload performance measurements, which are part of a confidential benchmarking
program. These recipes focus on helping you reliably achieve performance metrics,
such as throughput, that demonstrate the combined hardware and software stack on
TPUs.

**Note:** The recipes in this repository are not designed as general-purpose code
samples or tutorials for using Compute Engine-based products.

## Intended audience

This content is for you if you are a customer or partner who needs to:
- Validate hardware performance with your suppliers.
- Inform purchasing decisions using the benchmarking data.
- Reproduce optimal performance scenarios before you customize workflows for your
  own requirements.

## How to use these recipes

To reproduce a benchmark, follow these steps:

1.**Identify your requirements:** determine the model, TPU version, workload, and
  framework (JAX or PyTorch) that you are interested in.
2.**Select a recipe:** navigate to the appropriate directory, such as `./training`
  or `./inference`, to find a recipe that meets your needs.
3.**Follow the procedure:** each recipe guides you through preparing your environment,
  running the benchmark, and analyzing the results (including detailed logs). You can
  automate your infrastructure setup using Cluster Toolkit. For more information, see
  [Automated TPU environment deployment with Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview). 

## Repository organization

- `./training`: This directory contains recipes with instructions to reproduce the
  training performance of popular models, using PyTorch and JAX on specific TPU versions.
- `./inference`: This directory contains recipes that provide instructions and
  configurations to reproduce inference performance of models running on specific TPU
  versions.
- `./microbenchmarks`: This directory contains instructions for running low-level
  performance tests on TPUs, specifically focusing on matrix multiplication
  performance and memory bandwidth.
- `./utils`: This directory contains utility scripts for cluster and resource management
  for TPU7x (Ironwood) in GKE. For fully automated, production-ready cluster deployment,
  we recommend using the [Automated TPU environment deployment with Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-7x).

## XPK Recipes

The following table indexes all recipes and related documents that utilize [XPK](https://github.com/AI-Hypercomputer/xpk) for orchestrating and running workloads on GKE clusters.

| Recipe / File Name | Description | Link |
| :--- | :--- | :--- |
| Inference Wan2.1-T2V-14B on Ironwood | Pretrain  model in FP8 precision using XPK orchestration on Ironwood GKE clusters. | [README.md](inference/ironwood/MaxDiffusion/Wan2.x/Wan2.1-T2V/README.md) |
| Inference Wan2.2-T2V-27B on Ironwood | Pretrain  model in FP8 precision using XPK orchestration on Ironwood GKE clusters. | [README.md](inference/ironwood/MaxDiffusion/Wan2.x/Wan2.2-T2V/README.md) |
| Inference on Trillium | Run inference workloads on Trillium GKE clusters with XPK. | [README.md](inference/trillium/MaxDiffusion/Wan2.x/README.md) |
| Inference Wan2.2-T2V-27B on Trillium | Run Wan2.2-T2V-27B text-to-video diffusion inference workload on Trillium GKE clusters with XPK. | [README.md](inference/trillium/MaxDiffusion/Wan2.x/Wan2.2-T2V/README.md) |
| Collectives Benchmark on TPU Trillium | Run Collectives network performance microbenchmarks on TPU Trillium (v6e-256) GKE clusters using XPK. | [README.md](microbenchmarks/trillium/collectives/README.md) |
| Instructions for training Llama 3.0 70B on Trillium TPU on multipod using XPK | Train the Llama 3.0 70B PyTorch model on Trillium TPU clusters using multipod configuration with XPK. | [README.md](training/archive/trillium/Llama3.0-70B-PyTorch/XPK/README.md) |
| Instructions for training Llama 3.0 8B on Trillium TPU on multipod using XPK | Train the Llama 3.0 8B PyTorch model on Trillium TPU clusters using multipod configuration with XPK. | [README.md](training/archive/trillium/Llama3.0-8B-PyTorch/XPK/README.md) |
| Instructions for training Llama 3.1 405B on Trillium TPU on multipod using XPK | Train the Llama 3.1 405B PyTorch model on Trillium TPU clusters using multipod configuration with XPK. | [README.md](training/archive/trillium/Llama3.1-405B-PyTorch/XPK/README.md) |
| Instructions for training Mixtral 8x7B on Trillium TPU on multipod using XPK | Train the Mixtral 8x7B PyTorch model on Trillium TPU clusters using multipod configuration with XPK. | [README.md](training/archive/trillium/Mixtral-8x7B-Pytorch/XPK/README.md) |
| Pretrain DeepSeek-V3 671B BF16 (4k) on 4x4x8 | Pretrain DeepSeek-V3 671B model in BF16 precision on topology 4x4x8with 4k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x4x8/k8s/README.md) |
| Pretrain DeepSeek-V3 671B BF16 (4k) on 4x4x8 | Pretrain DeepSeek-V3 671B model in BF16 precision on topology 4x4x8with 4k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x4x8/xpk/README.md) |
| Pretrain DeepSeek-V3 671B BF16 (4k) on 4x8x8 with GCS | Pretrain DeepSeek-V3 671B model in BF16 precision on topology 4x8x8with 4k sequence length using Google Cloud Storage (GCS) using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x8x8-gcs/xpk/README.md) |
| Pretrain DeepSeek-V3 671B BF16 (4k) on 4x8x8 with Lustre | Pretrain DeepSeek-V3 671B model in BF16 precision on topology 4x8x8with 4k sequence length using Lustre using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x8x8-lustre/xpk/README.md) |
| Pretrain DeepSeek-V3 671B BF16 (4k) on 4x8x8 | Pretrain DeepSeek-V3 671B model in BF16 precision on topology 4x8x8with 4k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x8x8/xpk/README.md) |
| Pretrain DeepSeek-V3 671B FP8 (4k) on 4x4x8 | Pretrain DeepSeek-V3 671B model in FP8 precision on topology 4x4x8with 4k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/deepseek3-671b/4k-fp8-tpu7x-4x4x8/xpk/README.md) |
| Pretrain DeepSeek-V3 671B FP8 (4k) on 4x8x8 | Pretrain DeepSeek-V3 671B model in FP8 precision on topology 4x8x8with 4k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/deepseek3-671b/4k-fp8-tpu7x-4x8x8/xpk/README.md) |
| Pretrain Gemma4-26B BF16 (4k) on 4x4x4 | Pretrain Gemma4-26B model in BF16 precision on topology 4x4x4with 4k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/gemma4-26b/4k-bf16-tpu7x-4x4x4/xpk/README.md) |
| Pretrain Gemma4-26B BF16 (4k) on 4x8x8 | Pretrain Gemma4-26B model in BF16 precision on topology 4x8x8with 4k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/gemma4-26b/4k-bf16-tpu7x-4x8x8/xpk/README.md) |
| Pretrain Gemma4-2B BF16 (8k) on 4x4x4 | Pretrain Gemma4-2B model in BF16 precision on topology 4x4x4with 8k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/gemma4-2b/8k-bf16-tpu7x-4x4x4/xpk/README.md) |
| Pretrain Gemma4-31B BF16 (8k) on 4x4x4 | Pretrain Gemma4-31B model in BF16 precision on topology 4x4x4with 8k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/gemma4-31b/8k-bf16-tpu7x-4x4x4/xpk/README.md) |
| Pretrain Gemma4-4B BF16 (8k) on 4x4x4 | Pretrain Gemma4-4B model in BF16 precision on topology 4x4x4with 8k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/gemma4-4b/8k-bf16-tpu7x-4x4x4/xpk/README.md) |
| Pretrain GPT-OSS-120B BF16 (8k) on 4x4x4 | Pretrain GPT-OSS-120B model in BF16 precision on topology 4x4x4with 8k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/gpt-oss-120b/8k-bf16-tpu7x-4x4x4/xpk/README.md) |
| Pretrain GPT-OSS-120B BF16 (8k) on 4x8x8 | Pretrain GPT-OSS-120B model in BF16 precision on topology 4x8x8with 8k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/gpt-oss-120b/8k-bf16-tpu7x-4x8x8/xpk/README.md) |
| Pretrain Llama3.1-405B BF16 (8k) on 4x8x8 | Pretrain Llama3.1-405B model in BF16 precision on topology 4x8x8with 8k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/llama3.1-405b/8k-bf16-tpu7x-4x8x8/README.md) |
| Pretrain Llama3.1-405B FP8 (8k) on 4x8x8 | Pretrain Llama3.1-405B model in FP8 precision on topology 4x8x8with 8k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/llama3.1-405b/8k-fp8-tpu7x-4x8x8/README.md) |
| Pretrain Llama3.1-70B BF16 (128k) on 4x8x8 | Pretrain Llama3.1-70B model in BF16 precision on topology 4x8x8with 128k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/llama3.1-70b/128k-bf16-tpu7x-4x8x8/xpk/README.md) |
| Pretrain Llama3.1-70B FP8 (128k) on 4x8x8 | Pretrain Llama3.1-70B model in FP8 precision on topology 4x8x8with 128k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/llama3.1-70b/128k-fp8-tpu7x-4x8x8/README.md) |
| Pretrain Llama3.1-70B BF16 (8k) on 4x4x4 | Pretrain Llama3.1-70B model in BF16 precision on topology 4x4x4with 8k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/llama3.1-70b/8k-bf16-tpu7x-4x4x4/xpk/README.md) |
| Pretrain Llama3.1-70B BF16 (8k) on 4x8x8 | Pretrain Llama3.1-70B model in BF16 precision on topology 4x8x8with 8k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/llama3.1-70b/8k-bf16-tpu7x-4x8x8/xpk/README.md) |
| Pretrain Llama3.1-70B FP8 (8k) on 4x4x4 | Pretrain Llama3.1-70B model in FP8 precision on topology 4x4x4with 8k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/llama3.1-70b/8k-fp8-tpu7x-4x4x4/xpk/README.md) |
| Pretrain Llama3.1-70B FP8 (8k) on 4x8x8 | Pretrain Llama3.1-70B model in FP8 precision on topology 4x8x8with 8k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/llama3.1-70b/8k-fp8-tpu7x-4x8x8/README.md) |
| Pretrain Qwen3-235B BF16 (4k) on 4x8x8 | Pretrain Qwen3-235B model in BF16 precision on topology 4x8x8with 4k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/qwen3-235b-a22b/4k-bf16-tpu7x-4x8x8/xpk/README.md) |
| Pretrain Qwen3-235B FP8 (4k) on 4x8x8 | Pretrain Qwen3-235B model in FP8 precision on topology 4x8x8with 4k sequence length using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/qwen3-235b-a22b/4k-fp8-tpu7x-4x8x8/README.md) |
| Pretrain Wan2.1-14B BF16 on 4x4x4 | Pretrain Wan2.1-14B model in BF16 precision on topology 4x4x4 using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/wan2.1-14b/bf16-tpu7x-4x4x4/k8s/README.md) |
| Pretrain Wan2.1-14B BF16 on 4x4x4 | Pretrain Wan2.1-14B model in BF16 precision on topology 4x4x4 using XPK orchestration on Ironwood GKE clusters. | [README.md](training/ironwood/wan2.1-14b/bf16-tpu7x-4x4x4/xpk/README.md) |
| Train GPT3-175B in BF16 on TPU Trillium | Train GPT3-175B model on Trillium TPUs using MaxText in BF16 precision. | [README.md](training/trillium/GPT3-175B-MaxText/bf16/README.md) |
| Train GPT3-175B in FP8 on TPU Trillium | Train GPT3-175B model on Trillium TPUs using MaxText in FP8 precision. | [README.md](training/trillium/GPT3-175B-MaxText/fp8/README.md) |
| Train Gemma3-12B on TPU Trillium (2x v6e-256) | Train Gemma3-12B model on 2 slices of Trillium v6e-256 using MaxText. | [README.md](training/trillium/Gemma3-12B-MaxText/2x-v6e-256/README.md) |
| Train Gemma3-12B on TPU Trillium (4x v6e-256) | Train Gemma3-12B model on 4 slices of Trillium v6e-256 using MaxText. | [README.md](training/trillium/Gemma3-12B-MaxText/4x-v6e-256/README.md) |
| Train Gemma3-12B on TPU Trillium (v6e-256) | Train Gemma3-12B model on 1 slice of Trillium v6e-256 using MaxText. | [README.md](training/trillium/Gemma3-12B-MaxText/v6e-256/README.md) |
| Train Llama2-70B on TPU Trillium | Train Llama2-70B model on Trillium TPUs using MaxText. | [README.md](training/trillium/Llama2-70B-MaxText/README.md) |
| Train Llama3.1-405B on TPU Trillium | Train Llama3.1-405B model on Trillium TPUs using MaxText. | [README.md](training/trillium/Llama3.1-405B-MaxText/README.md) |
| Train Llama3.1-70B on TPU Trillium (v6e-128) | Train Llama3.1-70B model on Trillium TPUs (v6e-128) using MaxText. | [README.md](training/trillium/Llama3.1-70B-MaxText/v6e-128/README.md) |
| Train Llama3.1-70B on TPU Trillium (v6e-256) | Train Llama3.1-70B model on Trillium TPUs (v6e-256) using MaxText. | [README.md](training/trillium/Llama3.1-70B-MaxText/v6e-256/README.md) |
| Train Llama3.1-70B on TPU Trillium (v6e-32) | Train Llama3.1-70B model on Trillium TPUs (v6e-32) using MaxText. | [README.md](training/trillium/Llama3.1-70B-MaxText/v6e-32/README.md) |
| Train Llama3.1-70B on TPU Trillium (v6e-64) | Train Llama3.1-70B model on Trillium TPUs (v6e-64) using MaxText. | [README.md](training/trillium/Llama3.1-70B-MaxText/v6e-64/README.md) |
| Train Llama3.1-8B on TPU Trillium (v6e-128) | Train Llama3.1-8B model on Trillium TPUs (v6e-128) using MaxText. | [README.md](training/trillium/Llama3.1-8B-MaxText/v6e-128/README.md) |
| Train Llama3.1-8B on TPU Trillium (v6e-16) | Train Llama3.1-8B model on Trillium TPUs (v6e-16) using MaxText. | [README.md](training/trillium/Llama3.1-8B-MaxText/v6e-16/README.md) |
| Train Llama3.1-8B on TPU Trillium (v6e-256) | Train Llama3.1-8B model on Trillium TPUs (v6e-256) using MaxText. | [README.md](training/trillium/Llama3.1-8B-MaxText/v6e-256/README.md) |
| Train Llama3.1-8B on TPU Trillium (v6e-32) | Train Llama3.1-8B model on Trillium TPUs (v6e-32) using MaxText. | [README.md](training/trillium/Llama3.1-8B-MaxText/v6e-32/README.md) |
| Train Llama3.1-8B on TPU Trillium (v6e-64) | Train Llama3.1-8B model on Trillium TPUs (v6e-64) using MaxText. | [README.md](training/trillium/Llama3.1-8B-MaxText/v6e-64/README.md) |
| Train Llama3.1-8B on TPU Trillium (v6e-8) | Train Llama3.1-8B model on Trillium TPUs (v6e-8) using MaxText. | [README.md](training/trillium/Llama3.1-8B-MaxText/v6e-8/README.md) |
| Train Mistral-7B on TPU Trillium (v6e-8) | Train Mistral-7B model on Trillium TPUs (v6e-8) using MaxText. | [README.md](training/trillium/Mistral-7B-MaxText/README.md) |
| Train Mixtral-8x22B on TPU Trillium | Train Mixtral-8x22B model on Trillium TPUs using MaxText. | [README.md](training/trillium/Mixtral-8x22B-MaxText/README.md) |
| Train Mixtral-8x7B on TPU Trillium | Train the Mixtral-8x7B model on Trillium TPUs using MaxText. | [README.md](training/trillium/Mixtral-8x7B-MaxText/README.md) |
| Train DeepSeek3-671B on TPU v5p-1024 | Train DeepSeek-V3 671B mixture-of-experts model on TPU v5p-1024 clusters using MaxText. | [README.md](training/v5p/DeepSeek3-671B-MaxText/README.md) |
| Train Stable Diffusion 2 on TPU v5p | Train Stable Diffusion 2 on TPU v5p using the MaxDiffusion framework and XPK orchestration. | [README.md](training/v5p/Diffusion-2-MaxDiffusion/README.md) |
| Train GPT3-175B on TPU v5p | Train the GPT3-175B model on TPU v5p using MaxText. | [README.md](training/v5p/GPT3-175B-MaxText/README.md) |
| Train Llama2-7B on TPU v5p | Train the Llama2-7B model on TPU v5p using MaxText. | [README.md](training/v5p/Llama2-7B-Maxtext/README.md) |
| Train Llama3.1-405B on TPU v5p-1024 | Train the Llama3.1-405B model on TPU v5p-1024 clusters with MaxText and XPK. | [README.md](training/v5p/Llama3.1-405B-MaxText/README.md) |
| Train Llama4-Maverick-17B-128E on TPU v5p-256 | Train the Llama4-Maverick-17B-128E MoE model with MaxText on TPU v5p-256 slices. | [README.md](training/v5p/Llama4-Maverick-17B-128E-Maxtext/README.md) |
| Train Llama4-Scout-17B-16E on TPU v5p | Train the Llama4-Scout-17B-16E MoE model with MaxText on TPU v5p-256, v5p-512, or v5p-1024. | [README.md](training/v5p/Llama4-Scout-17B-16E-Maxtext/README.md) |
| Train Mixtral-8x7B on TPU v5p | Train the Mixtral-8x7B mixture-of-experts model on TPU v5p using MaxText. | [README.md](training/v5p/Mixtral-8X7B-Maxtext/README.md) |
| Train Stable Diffusion XL on TPU v5p | Train Stable Diffusion XL on TPU v5p using MaxDiffusion. | [README.md](training/v5p/SDXL-MaxDiffusion/README.md) |
| Pretrain Wan2.1-14B on TPU v5p-16 | Pretrain Wan2.1-14B MaxDiffusion model on v5p TPU GKE clusters using XPK. | [README.md](training/v5p/wan2.1-14b/bf16-v5p-16/xpk/README.md) |

## Repository scope

This repository provides the steps that you can use to reproduce a specific benchmark. 
The actual performance measurements and the complete, confidential benchmark report are 
not included.

## Methodology

Performance benchmarks measure the performance of various workloads on the platform. 
These benchmarks are primarily used to validate performance with hardware suppliers and 
to provide you with data for purchasing decisions.

## Maintenance policy

Benchmark data is considered a point-in-time measurement and completed benchmarks are not 
repeated. We maintain and update the recipes in this repository on a best-effort basis.

## Resources

For general guidance on using Google Cloud compute products, see the official documentation
and tutorials:

- [Compute Engine overview](https://docs.cloud.google.com/compute/docs/overview)
- [Compute Engine samples](https://docs.cloud.google.com/compute/docs/samples)
- [Cloud TPU documentation](https://docs.cloud.google.com/tpu/docs)
- [AI Hypercomputer documentation](https://docs.cloud.google.com/ai-hypercomputer/docs)
- [Automated TPU environment deployment with Cluster Toolkit](https://cloud.google.com/cluster-toolkit/docs/deploy/gke/gke-tpu-overview)

## Report issues

If you have questions or encounter problems with this repository, report them through
[GitHub Issues](https://github.com/AI-Hypercomputer/tpu-recipes/issues) or reach out to
your Google Cloud account team for assistance.

## Contributor notes

Note: This is not an officially supported Google product. This project is not eligible for
the  [Google Open Source Software Vulnerability Rewards Program](https://bughunters.google.com/open-source-security).
