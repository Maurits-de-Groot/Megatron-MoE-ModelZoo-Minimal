#!/bin/bash
set -euxo pipefail

# Path to Megatron-MoE-Scripts
export WORKSPACE=$(dirname "$(readlink -f "$0")")

# Benchmarking configurations (must be set)
export MODEL=${MODEL:-"Mixtral-8x22B"}
export CLUSTER=${CLUSTER:-"nebius_soperator"}
export MCORE_RELEASE_VERSION=${MCORE_RELEASE_VERSION:-"core_v0.13.0"} # Version and release info
export MEGATRON_PATH=${MEGATRON_PATH:-"${WORKSPACE}/Megatron-LM"} # Path to Megatron-LM
export CONTAINER_IMAGE=${CONTAINER_IMAGE:-"${WORKSPACE}/mg-modelzoo_pytorch+25.06.sqsh"} # Path to .sqsh or docker image url
export WANDB_API_KEY=${WANDB_API_KEY:-"key_here"} # Wandb API key
export NVIDIA_GDRCOPY=enabled

export CUDA_DEVICE_MAX_CONNECTIONS=1 #added

export PR="bf16"

# Load common configurations
source "${WORKSPACE}/runtime_configs/benchmarking/common.conf"
# Load model-specific configurations
source "${WORKSPACE}/runtime_configs/benchmarking/runtime.conf"
# Load cluster configurations
source "${WORKSPACE}/cluster_configs/benchmarking/${CLUSTER}.conf"

MORE_ARGS=(
    --no-mmap-bin-files
    --num-workers 8
# additional args
    --moe-token-dispatcher-type alltoall
    --overlap-grad-reduce
    --overlap-param-gather

    # --enable-cuda-graph
    # --te-rng-tracker
    # --moe-expert-capacity-factor 1.0
    # --moe-pad-expert-input-to-capacity
)

TRAINING_PARAMS="${MORE_ARGS[@]}"

# Initialize training parameters
TRAINING_PARAMS=${TRAINING_PARAMS:-""}

# Process training parameters
if [[ -f ${TRAINING_PARAMS_PATH} ]]; then
    envsubst < ${TRAINING_PARAMS_PATH} > ${TRAINING_PARAMS_PATH}.tmp
    TRAINING_PARAMS_PATH=${TRAINING_PARAMS_PATH}.tmp
else
    echo "Error: TRAINING_PARAMS_PATH does not exist: ${TRAINING_PARAMS_PATH}."
    exit 1
fi

# Extract training parameters to export
TRAINING_PARAMS_FROM_CONFIG=$(yq '... comments="" | .MODEL_ARGS | to_entries | .[] | 
    select(.value != "false") | 
    with(select(.value == "true"); .value = "") | 
    with(select(.key == "--pipeline-model-parallel-layout"); .value = (.value | @json)) | 
    [.key + " " + .value] | join("")' ${TRAINING_PARAMS_PATH} | tr '\n' ' ')
TRAINING_PARAMS="${TRAINING_PARAMS} ${TRAINING_PARAMS_FROM_CONFIG}"

# Append any command line arguments to TRAINING_PARAMS
if [[ $# -gt 0 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} $@"
fi

# Extract environment variables to export
ENV_VARS=$(yq '... comments="" | .ENV_VARS | to_entries | .[] | [.key + "=" + .value] | join(" ")' ${TRAINING_PARAMS_PATH})
while IFS='=' read -r KEY VALUE; do
    if [[ -n ${KEY} ]]; then
        export "${KEY}"="${VALUE}"
        echo "${KEY}=${VALUE}"
    fi
done < <(echo "${ENV_VARS}" | tr ' ' '\n')

# Virtual pipeline parallelism arguments
if [[ ${VPP} -gt 1 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --num-layers-per-virtual-pipeline-stage ${LAYERS_PER_VP}"
fi

# Uneven pipeline parallelism arguments
if [[ $((NUM_LAYERS % PP)) -ne 0 ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --decoder-first-pipeline-num-layers ${PP_FIRST} --decoder-last-pipeline-num-layers ${PP_LAST}"
fi

# FP8 arguments
if [[ ${PR} == "fp8" ]]; then
    TRAINING_PARAMS="${TRAINING_PARAMS} --fp8-recipe blockwise --fp8-format e4m3 --fp8-param-gather"
    TRAINING_PARAMS="${TRAINING_PARAMS} --use-precision-aware-optimizer --main-grads-dtype fp32 --main-params-dtype fp32 --exp-avg-dtype bf16 --exp-avg-sq-dtype bf16"
    TRAINING_PARAMS="${TRAINING_PARAMS} --moe-router-padding-for-fp8"
fi

# Profile command
if [[ ${PROFILE} -eq 1 ]]; then
    NSYS_PATH="${OUTPUT_PATH}/nsys"
    DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
    mkdir -p "${NSYS_PATH}"
    PROFILE_CMD="nsys profile --sample=none --cpuctxsw=none -t cuda,nvtx \
        --capture-range=cudaProfilerApi \
        --capture-range-end=stop \
        --cuda-memory-usage true \
        -f true -x true \
        -o ${NSYS_PATH}/${MODEL}-benchmarking-${DATETIME}"
    TRAINING_PARAMS="${TRAINING_PARAMS} --profile --profile-step-start 50 --profile-step-end 55 --profile-ranks 0 "
else
    PROFILE_CMD=""
fi

# Export training command
export TRAINING_CMD="${PROFILE_CMD} python ${TRAINING_SCRIPT_PATH} ${TRAINING_PARAMS}"

# SLURM settings
SLURM_LOGS="${OUTPUT_PATH}/slurm_logs"
mkdir -p ${SLURM_LOGS} || {
    echo "Error: Failed to create SLURM logs directory ${SLURM_LOGS}"
    exit 1
}

# Generate timestamp for job name
TIMESTAMP=$(date +'%y%m%d_%H%M%S')

# Submit SLURM job
set +e
sbatch <<EOF
#!/bin/bash

#SBATCH --nodes=${NNODES}
#SBATCH --account=${ACCOUNT}
#SBATCH --partition=${PARTITION}
#SBATCH --gpus-per-node=8
#SBATCH --mem=0
#SBATCH --time=${RUN_TIME}
#SBATCH --job-name=${ACCOUNT}-moe-${RUN_NAME}-${TIMESTAMP}
#SBATCH --dependency=singleton
#SBATCH --output=${WORKSPACE}/%x-%j.log
#SBATCH --exclusive

srun \
    --mpi=pmix -l \
    --ntasks-per-node=8 \
    --cpu-bind="verbose,mask_cpu:0x000000000000000000000000000000aa,0x00000000000000000000000000aa0000,0x0000000000000000000000aa00000000,0x000000000000000000aa000000000000,0x00000000000000aa0000000000000000,0x0000000000aa00000000000000000000,0x000000aa000000000000000000000000,0x00aa0000000000000000000000000000" \
    --mem-bind="verbose,map_mem:0,0,0,0,1,1,1,1" \
    --distribution=block:*:* \
    --container-image=${CONTAINER_IMAGE} \
    --container-mounts=${CONTAINER_MOUNTS} \
    --container-workdir=${MEGATRON_PATH} \
    bash -c \\\${TRAINING_CMD} 2>&1 | tee ${SLURM_LOGS}/\${SLURM_JOB_ID}.log
EOF
set -e