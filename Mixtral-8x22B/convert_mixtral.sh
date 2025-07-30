TOKENIZER_MODEL="${WORKDIR}/hf_checkpoint/tokenizer.model"
MEGATRON_PATH="${WORKDIR}/megatron-lm-convert"
export PYTHONPATH=$MEGATRON_PATH:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1

TARGET_TP_SIZE="2"
TARGET_EP_SIZE="8"
TARGET_PP_SIZE="8"

HF_FORMAT_DIR="${WORKDIR}/hf_checkpoint"
MEGATRON_FORMAT_DIR="${WORKDIR}/checkpoints/mixtral-mcore-TP${TARGET_TP_SIZE}PP${TARGET_PP_SIZE}EP${TARGET_EP_SIZE}"

torchrun --standalone --nnodes=1 --nproc_per_node=1 ${WORKDIR}/megatron-lm-convert/tools/checkpoint/convert.py \
--model-type GPT \
--loader mixtral_hf \
--saver mcore \
--target-tensor-parallel-size ${TARGET_TP_SIZE} \
--target-pipeline-parallel-size ${TARGET_PP_SIZE} \
--target-expert-parallel-size ${TARGET_EP_SIZE} \
--load-dir ${HF_FORMAT_DIR} \
--save-dir ${MEGATRON_FORMAT_DIR} \
--tokenizer-model ${TOKENIZER_MODEL}
