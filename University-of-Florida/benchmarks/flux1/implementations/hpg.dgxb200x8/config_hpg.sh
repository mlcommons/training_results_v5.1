# HPG Flags

export MLPERF_SUBMITTER="University of Florida"
export MLPERF_SYSTEM_NAME="HiPerGator NVIDIA DGX B200"
export NCCL_TEST=0
# export NCCL_NET_MERGE_LEVEL=LOC
# export CLEAR_CACHES=0

# Fixing HCA bug flags

export NCCL_IB_HCA=mlx5_15,mlx5_10,mlx5_14,mlx5_13,mlx5_8,mlx5_7,mlx5_9,mlx5_4
export NCCL_SOCKET_IFNAME=bridge-1145
export OMPI_MCA_pml=ucx
export OMPI_MCA_btl=^openib