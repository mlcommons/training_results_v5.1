#!/bin/bash

: "${WORKER_IDS:=0,1,2,3}"
: "${NEXP:=1}"
: "${GPU_ARCH:=b}"
: "${CPUS_PER_TASK:=20}"

echo "WORKER_IDS=${WORKER_IDS}"
echo "NEXP=${NEXP}"

export SLURM_MPI_TYPE=pmi2
export UCX_NET_DEVICES=mlx5_4:1,mlx5_5:1,mlx5_6:1,mlx5_7:1,mlx5_8:1,mlx5_9:1,mlx5_10:1,mlx5_11:1
export PMIX_GDS_MODULE=^ds12
export PMIX_MCA_gds=^ds12
export NCCL_NET_PLUGIN=none

export WORKDIR="/mnt/data/training_results_v5.1_Nebius-benchmarks.20251008/benchmarks/llama2_70b_lora"
export CONT="/mnt/data/images/20251008/llama2_70b_lora_20251008.sqsh"
export DATADIR="/mnt/data/datasets/llama2_70b_lora/gov_report"
export MODEL="/mnt/data/datasets/llama2_70b_lora/model"
export LOGDIR="/mnt/data/runs/b200n4/llama2_70b_lora/logs"
export CFGDIR="./"

source "${CFGDIR}/config_DGXB200_4x8x1xtp1pp1cp4_fp4.sh"
cd ${WORKDIR}
sbatch --job-name=${MODEL_SIZE} --exclusive \
  -N ${DGXNNODES} --gpus-per-node=${DGXNGPU} \
  --cpus-per-task ${CPUS_PER_TASK} --mem 0 \
  --nodelist="worker-[${WORKER_IDS}]" \
  --output="${LOGDIR}/slurm-%j.out" \
  run.sub

exit 0
