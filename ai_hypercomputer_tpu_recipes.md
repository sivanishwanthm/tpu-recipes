# AI Hypercomputer TPU Recipes Summary Table

This table provides a consolidated summary of configuration, benchmarking, and setup details for TPU recipes in AI Hypercomputer.

| Hardware | Model | Machine Type | Framework used | Workload Type | Link to the recipe (https) | Precision | Topology | Chips |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TPU | Wan2.1-T2V-14B | Ironwood (TPU v7x) | MaxDiffusion | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/MaxDiffusion/Wan2.x/Wan2.1-T2V/README.md | BF16 | N/A | N/A |
| TPU | Wan2.2-T2V-27B | Ironwood (TPU v7x) | MaxDiffusion | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/MaxDiffusion/Wan2.x/Wan2.2-T2V/README.md | BF16 | N/A | N/A |
| TPU | Gemma4 | Ironwood (TPU v7x) | vLLM | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Gemma4/README.md | FP8 | N/A | N/A |
| TPU | Qwen3-32B | Ironwood (TPU v7x) | vLLM | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Qwen3-32B/README.md | FP8 | N/A | N/A |
| TPU | Qwen3-Coder-480B-A35B | Ironwood (TPU v7x) | vLLM | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Qwen3-Coder-480B-A35B/README.md | FP8 | N/A | N/A |
| TPU | Qwen3-Embedding-8B | Ironwood (TPU v7x) | vLLM | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Qwen3-Embedding-8B/README.md | FP8 | N/A | N/A |
| TPU | Qwen3.5-397B | Ironwood (TPU v7x) | vLLM | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/ironwood/vLLM/Qwen3.5-397B/README.md | FP8 | N/A | N/A |
| TPU | Stable Diffusion | Trillium (v6e) | MaxDiffusion | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/MaxDiffusion/StableDiffusion/README.md | N/A | N/A | N/A |
| TPU | Wan2.2-T2V-27B | Trillium (v6e) | MaxDiffusion | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/MaxDiffusion/Wan2.x/Wan2.2-T2V/README.md | BF16 | N/A | N/A |
| TPU | Gemma4 | Trillium (v6e) | vLLM | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Gemma4/README.md | N/A | 2x4 | 8 |
| TPU | Llama3.x | Trillium (v6e) | vLLM | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Llama3.x/README.md | N/A | 2x4 | 8 |
| TPU | Qwen2.5-32B | Trillium (v6e) | vLLM | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Qwen2.5-32B/README.md | N/A | 2x2 | 4 |
| TPU | Qwen2.5-VL | Trillium (v6e) | vLLM | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Qwen2.5-VL/README.md | BF16 | 1x1 | 1 |
| TPU | Qwen3 | Trillium (v6e) | vLLM | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/trillium/vLLM/Qwen3/README.md | N/A | 2x2 | 4 |
| TPU | SDXL | v5e | MaxDiffusion | Inference | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/inference/v5e/MaxDiffusion/SDXL/README.md | N/A | N/A | N/A |
| TPU | N/A | Ironwood (TPU v7x) | N/A | Microbenchmark | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/microbenchmarks/ironwood/automation/autoscaling/README.md | N/A | N/A | N/A |
| TPU | N/A | Trillium (v6e) | N/A | Microbenchmark | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/microbenchmarks/trillium/collectives/README.md | BF16 | N/A | N/A |
| TPU | Stable Diffusion 2 | Trillium (v6e) | PyTorch/XLA | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/archive/trillium/Diffusion-2-PyTorch/README.md | BF16 | N/A | N/A |
| TPU | Llama3.0-70B | Trillium (v6e) | PyTorch/XLA | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/archive/trillium/Llama3.0-70B-PyTorch/GCE/README.md | BF16 | 16x16 | 256 |
| TPU | Llama3.0-70B | Trillium (v6e) | PyTorch/XLA | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/archive/trillium/Llama3.0-70B-PyTorch/XPK/README.md | BF16 | v6e-256 | 256 |
| TPU | Llama3.0-8B | Trillium (v6e) | PyTorch/XLA | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/archive/trillium/Llama3.0-8B-PyTorch/GCE/README.md | BF16 | 16x16 | 256 |
| TPU | Llama3.0-8B | Trillium (v6e) | PyTorch/XLA | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/archive/trillium/Llama3.0-8B-PyTorch/XPK/README.md | BF16 | v6e-256 | 256 |
| TPU | Llama3.1-405B | Trillium (v6e) | PyTorch/XLA | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/archive/trillium/Llama3.1-405B-PyTorch/GCE/README.md | BF16 | 16x16 | 256 |
| TPU | Llama3.1-405B | Trillium (v6e) | PyTorch/XLA | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/archive/trillium/Llama3.1-405B-PyTorch/XPK/README.md | BF16 | v6e-256 | 256 |
| TPU | Mixtral-8x7B | Trillium (v6e) | PyTorch/XLA | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/archive/trillium/Mixtral-8x7B-Pytorch/GCE/README.md | BF16 | N/A | N/A |
| TPU | Mixtral-8x7B | Trillium (v6e) | PyTorch/XLA | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/archive/trillium/Mixtral-8x7B-Pytorch/XPK/README.md | BF16 | v6e-256 | 256 |
| TPU | Stable Diffusion 2 | v5p | PyTorch/XLA | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/archive/v5p/Diffusion-2-PyTorch/README.md | BF16 | N/A | N/A |
| TPU | Llama2-7B | v5p | PyTorch/XLA | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/archive/v5p/Llama2-7B-PyTorch/README.md | BF16 | N/A | N/A |
| TPU | Mixtral-8x7B | v5p | PyTorch/XLA | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/archive/v5p/Mixtral-8x7B-PyTorch/README.md | BF16 | 4x4x8 | 128 |
| TPU | DeepSeek3-671B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x4x8/k8s/README.md | BF16 | 4x4x8 | 128 |
| TPU | DeepSeek3-671B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x4x8/xpk/README.md | BF16 | 4x4x8 | 128 |
| TPU | DeepSeek3-671B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x8x8-gcs/xpk/README.md | BF16 | 4x8x8 | 256 |
| TPU | DeepSeek3-671B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x8x8-lustre/xpk/README.md | BF16 | 4x8x8 | 256 |
| TPU | DeepSeek3-671B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x8x8/k8s/README.md | BF16 | 4x8x8 | 256 |
| TPU | DeepSeek3-671B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/deepseek3-671b/4k-bf16-tpu7x-4x8x8/xpk/README.md | BF16 | 4x8x8 | 256 |
| TPU | DeepSeek3-671B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/deepseek3-671b/4k-fp8-tpu7x-4x4x8/k8s/README.md | FP8 | 4x4x8 | 128 |
| TPU | DeepSeek3-671B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/deepseek3-671b/4k-fp8-tpu7x-4x4x8/xpk/README.md | FP8 | 4x4x8 | 128 |
| TPU | DeepSeek3-671B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/deepseek3-671b/4k-fp8-tpu7x-4x8x8/k8s/README.md | FP8 | 4x8x8 | 256 |
| TPU | DeepSeek3-671B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/deepseek3-671b/4k-fp8-tpu7x-4x8x8/xpk/README.md | FP8 | 4x8x8 | 256 |
| TPU | Gemma4-26B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gemma4-26b/4k-bf16-tpu7x-4x4x4/k8s/README.md | BFLOAT16 | 4x4x4 | 64 |
| TPU | Gemma4-26B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gemma4-26b/4k-bf16-tpu7x-4x4x4/xpk/README.md | BFLOAT16 | 4x4x4 | 64 |
| TPU | Gemma4-26B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gemma4-26b/4k-bf16-tpu7x-4x8x8/k8s/README.md | BFLOAT16 | 4x8x8 | 256 |
| TPU | Gemma4-26B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gemma4-26b/4k-bf16-tpu7x-4x8x8/xpk/README.md | BFLOAT16 | 4x8x8 | 256 |
| TPU | Gemma4-2B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gemma4-2b/8k-bf16-tpu7x-4x4x4/k8s/README.md | BFLOAT16 | 4x4x4 | 64 |
| TPU | Gemma4-2B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gemma4-2b/8k-bf16-tpu7x-4x4x4/xpk/README.md | BFLOAT16 | 4x4x4 | 64 |
| TPU | Gemma4-31B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gemma4-31b/8k-bf16-tpu7x-4x4x4/k8s/README.md | BFLOAT16 | 4x4x4 | 64 |
| TPU | Gemma4-31B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gemma4-31b/8k-bf16-tpu7x-4x4x4/xpk/README.md | BFLOAT16 | 4x4x4 | 64 |
| TPU | Gemma4-4B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gemma4-4b/8k-bf16-tpu7x-4x4x4/k8s/README.md | BFLOAT16 | 4x4x4 | 64 |
| TPU | Gemma4-4B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gemma4-4b/8k-bf16-tpu7x-4x4x4/xpk/README.md | BFLOAT16 | 4x4x4 | 64 |
| TPU | GPT-OSS-120B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gpt-oss-120b/8k-bf16-tpu7x-4x4x4/k8s/README.md | BF16 | 4x4x4 | 64 |
| TPU | GPT-OSS-120B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gpt-oss-120b/8k-bf16-tpu7x-4x4x4/xpk/README.md | BF16 | 4x4x4 | 64 |
| TPU | GPT-OSS-120B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gpt-oss-120b/8k-bf16-tpu7x-4x8x8/k8s/README.md | BF16 | 4x8x8 | 256 |
| TPU | GPT-OSS-120B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/gpt-oss-120b/8k-bf16-tpu7x-4x8x8/xpk/README.md | BF16 | 4x8x8 | 256 |
| TPU | Llama3.1-405B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/llama3.1-405b/8k-bf16-tpu7x-4x8x8/README.md | BF16 | 4x8x8 | 256 |
| TPU | Llama3.1-405B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/llama3.1-405b/8k-fp8-tpu7x-4x8x8/README.md | FP8 | 4x8x8 | 256 |
| TPU | Llama3.1-70B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/llama3.1-70b/128k-bf16-tpu7x-4x8x8/k8s/README.md | BF16 | 4x8x8 | 256 |
| TPU | Llama3.1-70B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/llama3.1-70b/128k-bf16-tpu7x-4x8x8/xpk/README.md | BF16 | 4x8x8 | 256 |
| TPU | Llama3.1-70B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/llama3.1-70b/128k-fp8-tpu7x-4x8x8/README.md | FP8 | 4x8x8 | 256 |
| TPU | Llama3.1-70B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/llama3.1-70b/8k-bf16-tpu7x-4x4x4/k8s/README.md | BF16 | 4x4x4 | 64 |
| TPU | Llama3.1-70B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/llama3.1-70b/8k-bf16-tpu7x-4x4x4/xpk/README.md | BF16 | 4x4x4 | 64 |
| TPU | Llama3.1-70B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/llama3.1-70b/8k-bf16-tpu7x-4x8x8/k8s/README.md | BF16 | 4x8x8 | 256 |
| TPU | Llama3.1-70B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/llama3.1-70b/8k-bf16-tpu7x-4x8x8/xpk/README.md | BF16 | 4x8x8 | 256 |
| TPU | Llama3.1-70B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/llama3.1-70b/8k-fp8-tpu7x-4x4x4/k8s/README.md | FP8 | 4x4x4 | 64 |
| TPU | Llama3.1-70B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/llama3.1-70b/8k-fp8-tpu7x-4x4x4/xpk/README.md | FP8 | 4x4x4 | 64 |
| TPU | Llama3.1-70B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/llama3.1-70b/8k-fp8-tpu7x-4x8x8/README.md | FP8 | 4x8x8 | 256 |
| TPU | Qwen3-235B-A22B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/qwen3-235b-a22b/4k-bf16-tpu7x-4x8x8/k8s/README.md | BF16 | 4x8x8 | 256 |
| TPU | Qwen3-235B-A22B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/qwen3-235b-a22b/4k-bf16-tpu7x-4x8x8/xpk/README.md | BF16 | 4x8x8 | 256 |
| TPU | Qwen3-235B-A22B | Ironwood (TPU v7x) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/qwen3-235b-a22b/4k-fp8-tpu7x-4x8x8/README.md | FP8 | 4x8x8 | 256 |
| TPU | Wan2.1-14B | Ironwood (TPU v7x) | MaxDiffusion | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/wan2.1-14b/bf16-tpu7x-4x4x4/k8s/README.md | BF16 | 4x4x4 | 64 |
| TPU | Wan2.1-14B | Ironwood (TPU v7x) | MaxDiffusion | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/ironwood/wan2.1-14b/bf16-tpu7x-4x4x4/xpk/README.md | BF16 | 4x4x4 | 64 |
| TPU | GPT3-175B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/GPT3-175B-MaxText/bf16/README.md | BF16 | N/A | N/A |
| TPU | GPT3-175B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/GPT3-175B-MaxText/fp8/README.md | FP8 | N/A | N/A |
| TPU | Gemma3-12B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Gemma3-12B-MaxText/2x-v6e-256/README.md | N/A | v6e-256 | 256 |
| TPU | Gemma3-12B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Gemma3-12B-MaxText/4x-v6e-256/README.md | N/A | v6e-256 | 256 |
| TPU | Gemma3-12B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Gemma3-12B-MaxText/v6e-256/README.md | N/A | v6e-256 | 256 |
| TPU | Llama2-70B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama2-70B-MaxText/README.md | N/A | N/A | N/A |
| TPU | Llama3.1-405B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama3.1-405B-MaxText/README.md | N/A | N/A | N/A |
| TPU | Llama3.1-70B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama3.1-70B-MaxText/v6e-128/README.md | N/A | v6e-128 | 128 |
| TPU | Llama3.1-70B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama3.1-70B-MaxText/v6e-256/README.md | N/A | v6e-256 | 256 |
| TPU | Llama3.1-70B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama3.1-70B-MaxText/v6e-32/README.md | BF16 | v6e-32 | 32 |
| TPU | Llama3.1-70B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama3.1-70B-MaxText/v6e-64/README.md | N/A | v6e-64 | 64 |
| TPU | Llama3.1-8B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama3.1-8B-MaxText/v6e-128/README.md | N/A | v6e-128 | 128 |
| TPU | Llama3.1-8B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama3.1-8B-MaxText/v6e-16/README.md | N/A | v6e-16 | 16 |
| TPU | Llama3.1-8B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama3.1-8B-MaxText/v6e-256/README.md | N/A | v6e-256 | 256 |
| TPU | Llama3.1-8B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama3.1-8B-MaxText/v6e-32/README.md | N/A | v6e-32 | 32 |
| TPU | Llama3.1-8B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama3.1-8B-MaxText/v6e-64/README.md | N/A | v6e-64 | 64 |
| TPU | Llama3.1-8B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Llama3.1-8B-MaxText/v6e-8/README.md | N/A | v6e-8 | 8 |
| TPU | Mistral-7B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Mistral-7B-MaxText/README.md | N/A | N/A | N/A |
| TPU | Mixtral-8x22B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Mixtral-8x22B-MaxText/README.md | BF16 | N/A | N/A |
| TPU | Mixtral-8x7B | Trillium (v6e) | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/trillium/Mixtral-8x7B-MaxText/README.md | N/A | N/A | N/A |
| TPU | DLRM-v2 | v5p | TensorFlow | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/DLRM-V2-Tensorflow/README.md | BF16 | N/A | N/A |
| TPU | DeepSeek3-671B | v5p | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/DeepSeek3-671B-MaxText/README.md | N/A | N/A | N/A |
| TPU | Stable Diffusion 2 | v5p | MaxDiffusion | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/Diffusion-2-MaxDiffusion/README.md | N/A | N/A | N/A |
| TPU | GPT3-175B | v5p | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/GPT3-175B-MaxText/README.md | N/A | N/A | N/A |
| TPU | Llama2-7B | v5p | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/Llama2-7B-Maxtext/README.md | N/A | N/A | N/A |
| TPU | Llama3.1-405B | v5p | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/Llama3.1-405B-MaxText/README.md | N/A | N/A | N/A |
| TPU | Llama4-Maverick-17B-128E | v5p | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/Llama4-Maverick-17B-128E-Maxtext/README.md | BF16 | N/A | N/A |
| TPU | Llama4-Scout-17B-16E | v5p | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/Llama4-Scout-17B-16E-Maxtext/README.md | N/A | N/A | N/A |
| TPU | Mixtral-8x7B | v5p | MaxText | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/Mixtral-8X7B-Maxtext/README.md | BF16 | N/A | N/A |
| TPU | SDXL | v5p | MaxDiffusion | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/SDXL-MaxDiffusion/README.md | BF16 | N/A | N/A |
| TPU | Wan2.1-14B | v5p | MaxDiffusion | Pre-training | https://github.com/AI-Hypercomputer/tpu-recipes/tree/main/training/v5p/wan2.1-14b/bf16-v5p-16/xpk/README.md | BF16 | N/A | N/A |
