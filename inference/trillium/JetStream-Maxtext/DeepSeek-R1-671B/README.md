# Inference benchmark of DeepSeek-R1-671B on Cloud TPU v6e

Step-by-step instructions on how to benchmark the inference of the DeepSeek-R1-671B model. This recipe uses JetStream with the MaxText engine on a Cloud TPU v6e (Trillium) VM.

## Prerequisites

- Your project has sufficient quota for a Cloud TPU v6e-64 slice and a 160-chip M1 machine.
- You have the following IAM roles on the project:
  - Compute Admin
  - Kubernetes Engine Admin
  - Storage Admin
  - Logging Admin
  - Monitoring Admin
  - Artifact Registry Writer
  - Service Account Admin
  - Project IAM Admin
- You have access to Pathways container images.
- You have a Hugging Face token with at least "Read" permissions to access the DeepSeek models.
- You have installed the following tools on your local environment:
  - [Google Cloud SDK](https://cloud.google.com/sdk/docs/install)
  - [Helm](https://helm.sh/docs/intro/install/)
  - [kubectl](https://kubernetes.io/docs/tasks/tools/install-kubectl-linux/)
  - [xpk](https://github.com/AI-Hypercomputer/xpk)

## Steps

### Step 1: Set up your local environment

1. Clone the `tpu-recipes` repository and set a reference to the recipe folder:

   ```bash
   git clone https://github.com/ai-hypercomputer/tpu-recipes.git
   cd tpu-recipes
   export REPO_ROOT=$(git rev-parse --show-toplevel)
   export RECIPE_ROOT=$REPO_ROOT/inference/trillium/JetStream-Maxtext/DeepSeek-R1-671B
   ```

2. Define the following environment variables with values appropriate for your workload:

   ```bash
   # Required variables to be set
   export PROJECT_ID=<PROJECT_ID>
   export REGION=<REGION>
   export CLUSTER_NAME=<CLUSTER_NAME>
   export CLUSTER_ZONE=<CLUSTER_ZONE>
   export GCS_BUCKET=<GCS_BUCKET>
   export TPU_RESERVATION=<TPU_RESERVATION>

   # Required variables with default values
   export TPU_TYPE=v6e-64
   export NUM_SLICES=1
   export CLUSTER_CPU_MACHINE_TYPE=n2d-standard-32
   export CLUSTER_CKPT_NODEPOOL_NAME=ckpt-conversion-node-pool-0
   export CLUSTER_CKPT_NODE_MACHINE_TYPE=m1-ultramem-160
   export CLUSTER_CKPT_NODE_REGION=us-east4
   export CLUSTER_CKPT_NODE_DISK_SIZE=3000
   export CLUSTER_CKPT_NUM_NODES=1
   export ARTIFACT_REGISTRY_REPO_NAME=jetstream-maxtext-ar
   export ARTIFACT_REGISTRY=${REGION}-docker.pkg.dev/${PROJECT_ID}/${ARTIFACT_REGISTRY_REPO_NAME}
   export JETSTREAM_MAXTEXT_IMAGE=jetstream-maxtext
   export JETSTREAM_MAXTEXT_VERSION=latest
   export HF_MODEL_NAME="deepseek-ai/DeepSeek-R1"
   export MODEL_NAME=deepseek3-671b
   export LOCAL_CKPT_BASE_PATH=/mnt/disks/persist
   export GCS_CKPT_PATH_BF16=gs://${GCS_BUCKET}/models/${MODEL_NAME}/bf16
   export GCS_CKPT_PATH_UNSCANNED=gs://${GCS_BUCKET}/models/${MODEL_NAME}/unscanned
   export GCS_CKPT_PATH_QUANTIZED=gs://${GCS_BUCKET}/models/${MODEL_NAME}/quantized
   ```

3. Set the default project:

   ```bash
   gcloud config set project $PROJECT_ID
   ```

### Step 2: Create a GKE cluster

1. Create a custom network:

   ```bash
   export NETWORK_NAME_1=${CLUSTER_NAME}-mtu9k-1
   export NETWORK_FW_NAME_1=${NETWORK_NAME_1}-fw-1
   gcloud compute networks create ${NETWORK_NAME_1} --mtu=8896 --project=${PROJECT_ID} --subnet-mode=auto --bgp-routing-mode=regional
   gcloud compute firewall-rules create ${NETWORK_FW_NAME_1} --network ${NETWORK_NAME_1} --allow tcp,icmp,udp --project=${PROJECT_ID}
   ```

2. Create a GKE cluster with a TPU v6e nodepool:

   ```bash
   export CLUSTER_ARGUMENTS="--enable-dataplane-v2 --enable-ip-alias --enable-multi-networking --network=${NETWORK_NAME_1} --subnetwork=${NETWORK_NAME_1} --scopes cloud-platform"
   export NODE_POOL_ARGUMENTS="--additional-node-network network=${NETWORK_NAME_2},subnetwork=${SUBNET_NAME_2} --scopes cloud-platform --workload-metadata=GCE_METADATA --placement-type=COMPACT"
   python3 ~/xpk/xpk.py cluster create \
     --cluster $CLUSTER_NAME \
     --default-pool-cpu-machine-type=$CLUSTER_CPU_MACHINE_TYPE \
     --num-slices=$NUM_SLICES \
     --tpu-type=$TPU_TYPE \
     --zone=${CLUSTER_ZONE} \
     --project=${PROJECT_ID} \
     --reservation=${TPU_RESERVATION} \
     --custom-cluster-arguments="${CLUSTER_ARGUMENTS}" \
     --custom-nodepool-arguments="${NODE_POOL_ARGUMENTS}"
   ```

### Step 3: Create a Cloud Storage bucket

Create a Cloud Storage bucket to store model checkpoints and temporary files:

```bash
gcloud storage buckets create gs://$GCS_BUCKET --location=$REGION
```

### Step 4: Configure a service account

Configure a Kubernetes service account to act as an IAM service account:

```bash
gcloud iam service-accounts create jetstream-pathways
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member "serviceAccount:jetstream-pathways@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role roles/storage.objectUser
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member "serviceAccount:jetstream-pathways@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role roles/storage.insightsCollectorService
kubectl annotate serviceaccount default \
  iam.gke.io/gcp-service-account=jetstream-pathways@${PROJECT_ID}.iam.gserviceaccount.com
```

### Step 5: Build the container image

1. Create an Artifact Registry repository:

   ```bash
   gcloud artifacts repositories create ${ARTIFACT_REGISTRY_REPO_NAME} \
         --repository-format=docker \
         --location=${REGION} \
         --description="Repository for JetStream/MaxText container images" \
         --project=${PROJECT_ID}
   ```

2. Configure Docker to authenticate to Artifact Registry:

   ```bash
   gcloud auth configure-docker ${REGION}-docker.pkg.dev
   ```

3. Build and push the Docker container image:

   ```bash
   cd $RECIPE_ROOT/docker
   gcloud builds submit \
     --project=${PROJECT_ID} \
     --region=${REGION} \
     --config cloudbuild.yml \
     --substitutions _ARTIFACT_REGISTRY=$ARTIFACT_REGISTRY,_JETSTREAM_MAXTEXT_IMAGE=$JETSTREAM_MAXTEXT_IMAGE,_JETSTREAM_MAXTEXT_VERSION=$JETSTREAM_MAXTEXT_VERSION \
     --timeout "2h" \
     --machine-type=e2-highcpu-32 \
     --disk-size=1000 \
     --quiet \
     --async
   ```

### Step 6: Convert the checkpoint

Submit a Cloud Batch job to download and convert the model checkpoint:

```bash
cd $RECIPE_ROOT/prepare-model
gcloud batch jobs submit convert-ckpt-to-unscanned-$(date +%Y%m%d-%H%M%S) \
  --project ${PROJECT_ID} \
  --location ${CLUSTER_CKPT_NODE_REGION} \
  --config - <<EOF
$(envsubst < batch_job.yaml)
EOF
```

### Step 7: Deploy JetStream and Pathways

1. Get cluster credentials:

   ```bash
   gcloud container clusters get-credentials $CLUSTER_NAME --region $REGION --project $PROJECT_ID
   ```

2. Create a Kubernetes secret with your Hugging Face token:

   ```bash
   export HF_TOKEN=<YOUR_HUGGINGFACE_TOKEN>
   kubectl create secret generic hf-secret \
     --from-literal=hf_api_token=${HF_TOKEN} \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

3. Deploy the LeaderWorkerSet (LWS) API:

   ```bash
   VERSION=v0.6.0
   kubectl apply --server-side -f "https://github.com/kubernetes-sigs/lws/releases/download/${VERSION}/manifests.yaml"
   ```

4. Deploy the workload manifest:

   ```bash
   cd $RECIPE_ROOT
   helm install -f values.yaml \
     --set volumes.gcsMounts[0].bucketName=${GCS_BUCKET} \
     --set clusterName=$CLUSTER_NAME \
     --set job.jax_tpu_image.repository=${ARTIFACT_REGISTRY}/${JETSTREAM_MAXTEXT_IMAGE} \
     --set job.jax_tpu_image.tag=${JETSTREAM_MAXTEXT_VERSION} \
     --set maxtext_config.load_parameters_path=${GCS_CKPT_PATH_QUANTIZED} \
     jetstream-pathways \
     $RECIPE_ROOT/serve-model
   ```

### Step 8: Run the MMLU benchmark

1. SSH into one of the `jax-tpu` workers:

   ```bash
   kubectl exec -it jetstream-pathways-0 -c jax-tpu -- /bin/bash
   ```

2. Download the MMLU dataset:

   ```bash
   LOCAL_DIR=/data
   mkdir -p ${LOCAL_DIR}/mmlu
   cd ${LOCAL_DIR}/mmlu
   wget https://people.eecs.berkeley.edu/~hendrycks/data.tar -P ${LOCAL_DIR}/mmlu
   tar -xvf data.tar
   ```

3. Run the benchmarking script:

   ```bash
   python3 /JetStream/benchmarks/benchmark_serving.py \
     --use-hf-tokenizer=True \
     --use-chat-template=False \
     --hf-access-token=$HF_TOKEN \
     --tokenizer=deepseek-ai/DeepSeek-R1 \
     --num-prompts 14037 \
     --dataset mmlu \
     --dataset-path ${LOCAL_DIR}/mmlu/data/test \
     --request-rate 0 \
     --warmup-mode sampled \
     --save-request-outputs \
     --num-shots=5 \
     --run-eval True \
     --model=deepseek3-671b \
     --save-result
   ```

### Step 9: Clean up

1. Delete the GKE cluster:

   ```bash
   gcloud container clusters delete $CLUSTER_NAME --zone $CLUSTER_ZONE
   ```

2. Delete the Cloud Storage bucket:

   ```bash
   gcloud storage buckets delete gs://${GCS_BUCKET}
   ```

3. Delete the VPC networks:

   ```bash
   # Delete resources for the second network
   gcloud compute routers nats delete ${NAT_CONFIG} --router=${ROUTER_NAME} --region=${REGION} --project=${PROJECT_ID} --quiet
   gcloud compute routers delete ${ROUTER_NAME} --region=${REGION} --project=${PROJECT_ID} --quiet
   gcloud compute firewall-rules delete ${FIREWALL_RULE_NAME} --project=${PROJECT_ID} --quiet
   gcloud compute networks subnets delete ${SUBNET_NAME_2} --region=${REGION} --project=${PROJECT_ID} --quiet
   gcloud compute networks delete ${NETWORK_NAME_2} --project=${PROJECT_ID} --quiet

   # Delete resources for the first network
   gcloud compute firewall-rules delete ${NETWORK_FW_NAME_1} --project=${PROJECT_ID} --quiet
   gcloud compute networks delete ${NETWORK_NAME_1} --project=${PROJECT_ID} --quiet
   ```
