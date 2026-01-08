# Inference benchmark of Llama 2 7B on Cloud TPU v6e

Step-by-step instructions on how to benchmark the inference of the Llama 2 7B model. This recipe uses JetStream with PyTorch on a Cloud TPU v6e (Trillium) VM.

## Prerequisites

- You have a Hugging Face token with at least "Read" permissions to access the Llama 2 model.

## Steps

### Step 1: Set up the environment

1. Create a virtual environment:

   ```bash
   export WORKDIR=$(pwd)
   cd $WORKDIR
   sudo apt install python3.10-venv
   python -m venv venv
   source venv/bin/activate
   ```

2. Clone the JetStream-PyTorch repository:

   ```bash
   git clone https://github.com/google/jetstream-pytorch.git
   cd jetstream-pytorch/
   git checkout jetstream-v0.2.4
   ```

3. Install dependencies:

   ```bash
   source install_everything.sh
   pip install -U --pre jax jaxlib libtpu-nightly requests -f https://storage.googleapis.com/jax-releases/jax_nightly_releases.html -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
   ```

4. Verify that Jax can access TPUs:

   ```bash
   python -c "import jax; print(jax.devices())"
   ```

### Step 2: Run the JetStream PyTorch server

1. Authenticate with Hugging Face:

   ```bash
   pip install -U "huggingface_hub[cli]"
   huggingface-cli login
   ```

2. Start the server:

   ```bash
   jpt serve --model_id meta-llama/Llama-2-7b-chat-hf
   ```

### Step 3: Send a request to the server

Send a gRPC request to the server to test the model:

```python
import requests
import os
import grpc

from jetstream.core.proto import jetstream_pb2
from jetstream.core.proto import jetstream_pb2_grpc

prompt = "What are the top 5 languages?"

channel = grpc.insecure_channel("localhost:9000")
stub = jetstream_pb2_grpc.OrchestratorStub(channel)

request = jetstream_pb2.DecodeRequest(
    text_content=jetstream_pb2.DecodeRequest.TextContent(
        text=prompt
    ),
    priority=0,
    max_tokens=2000,
)

response = stub.Decode(request)
output = []
for resp in response:
  output.extend(resp.stream_content.samples[0].text)

text_output = "".join(output)
print(f"Prompt: {prompt}")
print(f"Response: {text_output}")
```

### Step 4: Run the benchmark

1. In a new terminal, set up the benchmark environment:

   ```bash
   source venv/bin/activate
   cd jetstream-pytorch/deps/JetStream/benchmarks
   pip install -r requirements.in
   ```

2. Run the benchmark:

   ```bash
   export model_name=llama-2
   export tokenizer_path=../../../checkpoints/meta-llama/Llama-2-7b-chat-hf/hf_original/tokenizer.model
   python benchmark_serving.py --tokenizer $tokenizer_path --num-prompts 1000  --dataset openorca --save-request-outputs --warmup-mode=sampled --model=$model_name
   ```
