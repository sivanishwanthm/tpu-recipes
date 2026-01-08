# Serve GPT-OSS 120B with vLLM on Cloud TPU v7

Step-by-step instructions on how to serve the GPT-OSS 120B model on a Cloud TPU v7 (Ironwood) VM. This recipe uses vLLM on Google Kubernetes Engine (GKE).

## Prerequisites

- You have installed the [gcloud CLI](https://cloud.google.com/sdk/docs/install#mac).
- You have a GKE cluster configured with the necessary networking and identity features.
- You have a Hugging Face token with at least "Read" permissions.

## Steps

### Step 1: Define environment variables

Define the following environment variables with values appropriate for your workload:

```bash
# Set variables if not already set
export CLUSTER_NAME=<YOUR_CLUSTER_NAME>
export PROJECT_ID=<YOUR_PROJECT_ID>
export REGION=<YOUR_REGION>
export ZONE=<YOUR_ZONE> # e.g., us-central1-a
export NODEPOOL_NAME=<YOUR_NODEPOOL_NAME>
export RESERVATION_NAME=<YOUR_RESERVATION_NAME> # Optional, if you have a reservation
```

### Step 2: Create a GKE cluster

Create a GKE cluster with the required features enabled:

```bash
gcloud container clusters create $CLUSTER_NAME \
  --project=$PROJECT_ID \
  --location=$REGION \
  --workload-pool=$PROJECT_ID.svc.id.goog \
  --release-channel=rapid \
  --num-nodes=1 \
  --gateway-api=standard \
  --addons GcsFuseCsiDriver,HttpLoadBalancing
```

### Step 3: Create a proxy-only subnet

A proxy-only subnet is required for the regional managed proxy:

```bash
# Set variables if not already set
export PROXY_ONLY_SUBNET_NAME="tpu-proxy-subnet"
export VPC_NETWORK_NAME="default" # Or your custom VPC
export PROXY_ONLY_SUBNET_RANGE="10.129.0.0/23" # Example range

gcloud compute networks subnets create ${PROXY_ONLY_SUBNET_NAME} \
  --purpose=REGIONAL_MANAGED_PROXY \
  --role=ACTIVE \
  --region ${REGION} \
  --network "${VPC_NETWORK_NAME}" \
  --range "${PROXY_ONLY_SUBNET_RANGE}" \
  --project ${PROJECT_ID}
```

### Step 4: Create a nodepool

Create a nodepool with a single TPU v7 node in a 2x2x1 configuration:

```bash
gcloud container node-pools create ${NODEPOOL_NAME} \
  --project=${PROJECT_ID} \
  --location=${REGION} \
  --node-locations=${ZONE} \
  --num-nodes=1 \
  --reservation=${RESERVATION_NAME} \
  --reservation-affinity=specific \
  --machine-type=tpu7x-standard-4t \
  --cluster=${CLUSTER_NAME}
```

### Step 5: Deploy the vLLM workload on GKE

1. Configure kubectl to communicate with your cluster:

   ```bash
   gcloud container clusters get-credentials ${CLUSTER_NAME} --location=us-central1-c
   ```

2. Create a Kubernetes secret for your Hugging Face credentials:

   ```bash
   export HF_TOKEN=YOUR_TOKEN
   kubectl create secret generic hf-secret \
       --from-literal=hf_api_token=${HF_TOKEN}
   ```

3. Save the following YAML file as `vllm-tpu.yaml`:

   ```yaml
   apiVersion: storage.k8s.io/v1
   kind: StorageClass
   metadata:
     name: hyperdisk-balanced-tpu
   provisioner: pd.csi.storage.gke.io
   parameters:
     type: hyperdisk-balanced
   reclaimPolicy: Delete
   volumeBindingMode: WaitForFirstConsumer
   allowVolumeExpansion: true
   ---
   apiVersion: v1
   kind: PersistentVolumeClaim
   metadata:
     name: hd-claim
   spec:
     storageClassName: hyperdisk-balanced-tpu
     accessModes:
       - ReadWriteOnce
     resources:
       requests:
         storage: 200Gi
   ---
   apiVersion: apps/v1
   kind: Deployment
   metadata:
     name: vllm-tpu
   spec:
     replicas: 1
     selector:
       matchLabels:
         app: vllm-tpu
     template:
       metadata:
         labels:
           app: vllm-tpu
       spec:
         nodeSelector:
           cloud.google.com/gke-tpu-accelerator: tpu7x
           cloud.google.com/gke-tpu-topology: 2x2x1
         containers:
         - name: vllm-tpu
           image: vllm/vllm-tpu:nightly-ironwood-20251217-baf570b-0cd5353
           command: ["python3", "-m", "vllm.entrypoints.openai.api_server"]
           args:
           - --host=0.0.0.0
           - --port=8000
           - --tensor-parallel-size=2
           - --data-parallel-size=4
           - --max-model-len=9216
           - --download-dir=/data
           - --max-num-batched-tokens=16384
           - --max-num-seqs=2048
           - --no-enable-prefix-caching
           - --model=openai/gpt-oss-120b
           - --kv-cache-dtype=fp8
           - --async-scheduling
           - --gpu-memory-utilization=0.93
           env:
           - name: HF_HOME
             value: /data
           - name: HUGGING_FACE_HUB_TOKEN
             valueFrom:
               secretKeyRef:
                 name: hf-secret
                 key: hf_api_token
           - name: TPU_BACKEND_TYPE
             value: jax
           - name: MODEL_IMPL_TYPE
             value: vllm
           ports:
           - containerPort: 8000
           resources:
             limits:
               google.com/tpu: '4'
             requests:
               google.com/tpu: '4'
           readinessProbe:
             tcpSocket:
               port: 8000
             initialDelaySeconds: 15
             periodSeconds: 10
           volumeMounts:
           - mountPath: "/data"
             name: data-volume
           - mountPath: /dev/shm
             name: dshm
         volumes:
         - emptyDir:
             medium: Memory
           name: dshm
         - name: data-volume
           persistentVolumeClaim:
             claimName: hd-claim
   ---
   apiVersion: v1
   kind: Service
   metadata:
     name: vllm-service
   spec:
     selector:
       app: vllm-tpu
     type: LoadBalancer
     ports:
       - name: http
         protocol: TCP
         port: 8000
         targetPort: 8000
   ```

4. Apply the vLLM manifest:

   ```bash
   kubectl apply -f vllm-tpu.yaml
   ```

### Step 6: Test the model

1. Port-forward the service:

   ```bash
   kubectl port-forward service/vllm-service 8000:8000
   ```

2. Interact with the model using `curl`:

   ```bash
   curl http://localhost:8000/v1/completions -H "Content-Type: application/json" -d '{
       "model": "openai/gpt-oss-120b",
       "prompt": "San Francisco is a",
       "max_tokens": 7,
       "temperature": 0
   }'
   ```
