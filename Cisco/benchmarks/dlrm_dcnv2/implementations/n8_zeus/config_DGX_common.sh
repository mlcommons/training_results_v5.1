#Performance paramenter
export DLRM_BIND="numactl --membind=0,1"
export NCCL_IB_GID_INDEX=3
export NCCL_TEST=0

export NCCL_SOCKET_IFNAME="gpu0_eth,gpu1_eth,gpu2_eth,gpu3_eth,gpu4_eth,gpu5_eth,gpu6_eth,gpu7_eth"
export UCX_NET_DEVICES="gpu0_eth,gpu1_eth,gpu2_eth,gpu3_eth,gpu4_eth,gpu5_eth,gpu6_eth,gpu7_eth"
export NCCL_IB_HCA="gpu0_rdma,gpu1_rdma,gpu2_rdma,gpu3_rdma,gpu4_rdma,gpu5_rdma,gpu6_rdma,gpu7_rdma"
