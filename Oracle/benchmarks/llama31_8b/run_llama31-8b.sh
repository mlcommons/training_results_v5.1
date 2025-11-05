#!/bin/bash

BASE_DIR=/mnt/lfs/sce/mlperf_train_v51/llama31_8b/b200
export DATADIR=$BASE_DIR/data
export LOGDIR=$BASE_DIR/logs
export CONT=$BASE_DIR/llama3_8b_20251008.sqsh
#source config_DGXB200_1x8x2xtp1pp1cp1_8b.sh
#source config_DGXB200_1x8x2xtp1pp1cp1_8b_fp4.sh
source config_DGXB200_8x8x1xtp1pp1cp2_8b.sh
#source config_DGXB200_9x8x1xtp1pp1cp1_8b_fp4.sh
#source config_DGXB200_16x8x1xtp1pp1cp2_8b_fp4.sh

#---------------------- OCI Extras -------------------
export OMPI_MCA_btl_tcp_if_include="10.224.0.0/12"
export PMIX_MCA_gds="^ds12"
export NCCL_DEBUG=WARN
export NCCL_CUMEM_ENABLE=0
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_QPS_PER_CONNECTION=2
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TC=41
export NCCL_IB_SL=0
export NCCL_IB_TIMEOUT=22
export NCCL_NET_PLUGIN=none
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IGNORE_CPU_AFFINITY=1
export RX_QUEUE_LEN=8192
export IB_RX_QUEUE_LEN=8192
export UCX_NET_DEVICES=eth0
export UCX_TLS=tcp
export HCOLL_ENABLE_MCAST_ALL=0
export coll_hcoll_enable=0
export NCCL_IB_HCA='=mlx5_0,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_9,mlx5_10,mlx5_11'

#----------------------- Launch Job ------------------
export SLURM_MPI_TYPE=pmi2

sbatch --job-name=llama31-8b \
       	--exclusive \
       	--gpus-per-node=8 \
       	-N ${DGXNNODES} \
	--time=${WALLTIME} \
       	run.sub

#--cpu-bind=verbose,mask_cpu:"0x0000000000000000000000003fff,0x000000000000000000000fffc000,0x000000000000000003fff0000000,0x00000000000000fffc0000000000,0x00000000003fff00000000000000,0x0000000fffc00000000000000000,0x0003fff000000000000000000000,0xfffc000000000000000000000000"
