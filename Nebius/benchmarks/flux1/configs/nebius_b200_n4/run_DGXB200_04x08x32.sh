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
export JET=0

export WORKDIR="/mnt/data/training_results_v5.1_Nebius-benchmarks.20251004/benchmarks/flux1"
export CONT="/mnt/data/images/20251004/flux1_20251004.sqsh"
export DATADIR="/mnt/data/datasets/flux1"
export CFGDIR="./"
export LOGDIR="/mnt/data/runs/b200n4/flux1/logs"
export DATAROOT="${DATADIR}/energon"

source "${CFGDIR}/config_DGXB200_04x08x32.sh"
cd ${WORKDIR}
sbatch --job-name=flux1 --exclusive \
  -N ${DGXNNODES} --gpus-per-node=${DGXNGPU} \
  --cpus-per-task ${CPUS_PER_TASK} --mem 0 \
  --nodelist="worker-[${WORKER_IDS}]" \
  --output="${LOGDIR}/slurm-%j.out" \
  run.sub

exit 0
