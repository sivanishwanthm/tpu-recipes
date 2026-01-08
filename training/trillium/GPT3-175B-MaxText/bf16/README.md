# Train GPT-3 175B on Cloud TPU v6e

Step-by-step instructions on how to train the GPT-3 175B model. This recipe uses MaxText on a Cloud TPU v6e (Trillium) VM.

## Prerequisites

- You have a GKE cluster created with [XPK](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/XPK_README.md).
- You have installed MaxText and built the Docker image as described in the [MAXTEXT_README](https://github.com/AI-Hypercomputer/tpu-recipes/blob/main/training/MAXTEXT_README.md).

## Steps

### Step 1: Set up the environment

1. When installing MaxText, use the `tpu-recipes-v0.1.2` tag:

   ```bash
   git checkout tpu-recipes-v0.1.2
   ```

2. When building the Docker image, use the `jax-stable-stack` image:

   ```bash
   BASE_IMAGE=us-docker.pkg.dev/cloud-tpu-images/jax-stable-stack/tpu:jax0.5.2-rev1
   bash docker_build_dependency_image.sh DEVICE=tpu MODE=stable_stack BASEIMAGE=${BASE_IMAGE}
   ```

### Step 2: Run the workload

From the MaxText root directory, start the GPT-3 175B workload:

```bash
python3 -m benchmarks.benchmark_runner xpk \
    --project=$PROJECT \
    --zone=$ZONE \
    --device_type=v6e-256 \
    --num_slices=1  \
    --cluster_name=${CLUSTER_NAME}  \
    --base_output_directory=${OUTPUT_DIR} \
    --model_name="gpt_3_175b_bf16" \
    --base_docker_image=maxtext_base_image
```
