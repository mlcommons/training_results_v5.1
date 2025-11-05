
export TRAIN_ONLY=0

export USE_DIST_OPTIMIZER=True

export NVTE_FWD_LAYERNORM_SM_MARGIN=8
export NVTE_BWD_LAYERNORM_SM_MARGIN=8

export NCCL_MIN_NCHANNELS=4

export CUDA_DEVICE_MAX_CONNECTIONS=1

export MICRO_BATCH_SIZE=1

: "${LOAD_MINIMAL_NUM_SAMPLES:=0}"

if [[ "${LOAD_MINIMAL_NUM_SAMPLES}" -eq 1 ]]; then
  export MAX_STEPS=500
  export OVERRIDE_ZERO_CONSUMED_SAMPLES=0
  export INIT_GLOBAL_STEP=0
fi

# # This is needed to save memory. nvbug 4264087 tracks
# export NCCL_NVLS_ENABLE=0

export TE_UB_ATOMIC_GEMM_RS=0
export MC_TP_OVERLAP_AG=True
export MC_TP_OVERLAP_RS=True

#TODO: remove this line when TP overlap works properly for eval
export NVTE_TP_OVERLAP_TRAINING_ONLY=1

# FA: Disbale FAv2 from cuDNN and optimizations that consume memory (expected < 200MB) as they cause IMAs
#export NVTE_FUSED_ATTN=0 # Disable cuDNN fused attention
#export NVTE_FUSED_ATTN_FORCE_WORKSPACE_OPT=0

export NCCL_P2P_NET_CHUNKSIZE=2097152

# Enable per-communicator nccl option tuning
export NCCL_CFG_PATH="/workspace/llm/conf/nccl/custom_communicator_cta.yaml"

# Disable gc when switching to/from validation steps
export NEMO_MANUAL_GC_IN_VALIDATION=0

# skip unnecessary broadcasting of training loss
export NEMO_LOG_TRAIN_LOSS=1

export NCCL_SHARP_GROUP_SIZE_THRESH=2  #Avoid falling back to non-sharp

export FP8=True

export NCCL_WORK_FIFO_DEPTH=1048576

# Use legacy NeMo dataset path
export LEGACY_DATASET=True

if [[ "${NO_CKPT:-0}" -eq 1 ]]; then
    export LOAD_CHECKPOINT=""
    export CHECK_COMPLIANCE="0"
fi

export DEFER_EMBEDDING_WGRAD_COMPUTE=True
export WGRAD_DEFERRAL_LIMIT=50

export OVERLAP_GRAD_REDUCE=True
export OVERLAP_PARAM_GATHER=True
export ALIGN_PARAM_GATHER=True
export OVERLAP_PARAM_GATHER_WITH_OPTIM_STEP=True
export FP8_PARAMS=True

#To silent warnings that print during training
export TOKENIZERS_PARALLELISM=False

export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

