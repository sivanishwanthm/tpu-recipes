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

1. **Identify your requirements:** determine the model, TPU version, workload, and
   framework (JAX or PyTorch) that you are interested in.
2. **Select a recipe:** navigate to the appropriate directory, such as `./training`
   or `./inference`, to find a recipe that meets your needs.
3. **Follow the procedure:** each recipe guides you through preparing your environment,
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

---

## Inference Recipes Summary

The table below summarizes all inference recipes under `inference/` across Cloud TPU hardware generations (`v6e` / Trillium and `v7x` / Ironwood), categorized by architecture type, software stack, orchestration environment, host type, models used, and Customer User Journey (CUJ).

| Type | Accelerator | Orchestrator | Software Stack | Host Type | Models Used | CUJ | Link to the recipe |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Diffusion | v6e | GKE & GCE | MaxDiffusion | single host | SDXL, SD 2.1 | CUJ 7: Serve a Diffusion model (E.g: WAN 2.1 or SDXL) on GKE Trillium using MaxDiffusion | [inference/trillium/MaxDiffusion/StableDiffusion](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/MaxDiffusion/StableDiffusion) |
| Diffusion | v6e | GKE & GCE | MaxDiffusion | single host | WAN 2.1 (T2V 1.3B) | CUJ 7: Serve a Diffusion model (E.g: WAN 2.1 or SDXL) on GKE Trillium using MaxDiffusion | [inference/trillium/MaxDiffusion/Wan2.x](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/MaxDiffusion/Wan2.x) |
| Diffusion | v6e | GKE & GCE | MaxDiffusion | single host | WAN 2.2-T2V (1.3B) | CUJ 8: Serve a Diffusion model (E.g: WAN 2.2- T2V) on GKE Trillium using MaxDiffusion | [inference/trillium/MaxDiffusion/Wan2.x/Wan2.2-T2V](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/MaxDiffusion/Wan2.x/Wan2.2-T2V) |
| MOE & Dense | v6e | GCE | vLLM | single host | Gemma 4 31B IT, Gemma 4 26B-A4B IT | Serve a multimodal dense/MoE model (Gemma 4 31B/26B) on GCE Trillium using vLLM single host | [inference/trillium/vLLM/Gemma4](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Gemma4) |
| Dense | v6e | GCE | vLLM | multihost | Llama 3.1 8B, Llama 3.3 70B | CUJ 6: Serve a dense OSS model (E.g: Llama 70B or Qwen3 -32B) using multihost serving on GKE using TPU v6e and vLLM | [inference/trillium/vLLM/Llama3.x](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Llama3.x) |
| Dense | v6e | GCE | vLLM | single host | Qwen2.5-32B | Serve a dense OSS model (Qwen 2.5 32B) on GCE Trillium using vLLM single host | [inference/trillium/vLLM/Qwen2.5-32B](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Qwen2.5-32B) |
| Dense | v6e | GCE | vLLM | single host | Qwen2.5-VL-7B | Serve a multimodal dense model (Qwen2.5-VL-7B) on GCE Trillium using vLLM single host | [inference/trillium/vLLM/Qwen2.5-VL](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Qwen2.5-VL) |
| Dense | v6e | GCE | vLLM | single host | Qwen3-4B, Qwen3-32B | Serve a dense OSS model (Qwen3 4B/32B) on GCE Trillium using vLLM single host | [inference/trillium/vLLM/Qwen3](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Qwen3) |
| Diffusion | v7x | GKE & GCE | MaxDiffusion | single host | WAN 2.1 (T2V 1.3B) | Serve a Diffusion model (WAN 2.1) on GKE/GCE Ironwood using MaxDiffusion | [inference/ironwood/MaxDiffusion/Wan2.x/Wan2.1-T2V](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/MaxDiffusion/Wan2.x/Wan2.1-T2V) |
| Diffusion | v7x | GKE & GCE | MaxDiffusion | single host | WAN 2.2-T2V (1.3B) | Serve a Diffusion model (WAN 2.2-T2V) on GKE/GCE Ironwood using MaxDiffusion | [inference/ironwood/MaxDiffusion/Wan2.x/Wan2.2-T2V](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/MaxDiffusion/Wan2.x/Wan2.2-T2V) |
| MOE & Dense | v7x | GKE | vLLM | single host | Gemma 4 31B IT, Gemma 4 26B-A4B IT | CUJ 1: Serve a multimodal MoE model (E.g: Gemma 4 - 26B) on GKE Ironwood using vLLM single host | [inference/ironwood/vLLM/Gemma4](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Gemma4) |
| Dense | v7x | GKE | vLLM | single host | GPT-OSS 120B | Serve a dense OSS model (GPT-OSS 120B) on GKE Ironwood using vLLM single host | [inference/ironwood/vLLM/GPT-OSS](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/GPT-OSS) |
| Dense | v7x | GKE | vLLM | single host | Qwen3-32B | CUJ 4: Serve a dense OSS model (e.g: Qwen 3 - 32B or Gemma 4- 31B) on GKE Ironwood using vLLM single host | [inference/ironwood/vLLM/Qwen3-32B](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Qwen3-32B) |
| MOE | v7x | GKE | vLLM | multihost | Qwen3-Coder-480B-A35B | CUJ 2: Serve a large multimodal MoE model (E.g: DeepSeek v3/R1 671B, Qwen 3.5- 397B or Qwen 3- Coder - 480B) on GKE/ Ironwood using vLLM as multihost | [inference/ironwood/vLLM/Qwen3-Coder-480B-A35B](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Qwen3-Coder-480B-A35B) |
| Dense | v7x | GCE | vLLM | single host | Qwen3-Embedding-8B | Serve a dense embedding model (Qwen3-Embedding-8B) on GCE Ironwood using vLLM single host | [inference/ironwood/vLLM/Qwen3-Embedding-8B](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Qwen3-Embedding-8B) |
| Dense | v7x | GCE | vLLM | single host | Qwen3-VL-Embedding-8B | Serve a multimodal dense embedding model (Qwen3-VL-Embedding-8B) on GCE Ironwood using vLLM single host | [inference/ironwood/vLLM/Qwen3-VL-Embedding-8B](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Qwen3-VL-Embedding-8B) |
| MOE | v7x | GKE | vLLM | multihost | Qwen3.5-397B | CUJ 2: Serve a large multimodal MoE model (E.g: DeepSeek v3/R1 671B, Qwen 3.5- 397B or Qwen 3- Coder - 480B) on GKE/ Ironwood using vLLM as multihost | [inference/ironwood/vLLM/Qwen3.5-397B](https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Qwen3.5-397B) |

---

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
