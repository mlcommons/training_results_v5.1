# Supermicro_MangoBoost MLPerf Training v5.1 Benchmark

This folder contains all of the code necessary to run:

+ MLPerf Training MI325X Single-Node (8 GPUs on a single node)

---

# 1. Prerequisites

- **Hardware:** AMD MI325X or compatible GPUs, supported RDMA/NICs, and sufficient system resources.
- **Software:** 
  - Docker (with permission to run as your user)
  - ROCm stack and drivers installed and loaded
  - `cpupower`, `rocm_agent_enumerator`, and other system utilities available
  - Access to the required MLPerf data and model directories
- **Permissions:** Ability to run `sudo` for certain system tuning commands (e.g., dropping caches, setting kernel parameters).

---

**System Tuning (Recommended by AMD, for AMD Instinct MI300X systems):**

Before running the benchmark, you must run the system tuning script once per node after each system restart. This script applies recommended kernel and CPU settings for optimal ROCm/MI300X performance. To find out more, please review [AMD's optimization manual](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/system-optimization/mi300x.html#mi300x-os-settings).

```bash
bash runtime_tunables.sh
```

> **Note:** This script must be run with appropriate privileges (may require `sudo` for some commands). Run it **once per node after each reboot**, before launching `docker_run.sh`.

---

# 2. Environment Setup

Define or `source` the following **required** variables on each node.

```bash
export RUN_DATETIME=`date +%Y%m%d-%H%M%S`
export DGXSYSTEM="MI325X"
export DATADIR="/path/to/llama2/70b/data"
export MODELDIR="/path/to/llama2/70b/model"
export IMAGE_NAME="llmboost/mb-llmboost-training:mlperf-5.1-prod"
export NCCL_SOCKET_IFNAME="comma,separated,list,of,RDMA-enabled,interfaces"
export GLOO_SOCKET_IFNAME="comma,separated,list,of,ethernet,interfaces"
export MASTER_ADDR="master-node.org.infra"
export MASTER_PORT=29500
export WORLD_SIZE= # Num. of nodes: [ 1 | 2 | 4 | ... ]
export RANK= # Rank of a given node: [ 0 | 1 | ... ]
export NCCL_DEBUG="" # NCCL log level: [ VERSION | WARN | INFO ]
export CONTAINER="mb_llmboost_llama2_${RUN_DATETIME}_${RANK}"
```

| Variable             | Description                                                        | Example Value                                   |
| -------------------- | ------------------------------------------------------------------ | ----------------------------------------------- |
| `RUN_DATETIME`       | Unique timestamp for this run (recommended: `date +%Y%m%d-%H%M%S`) | `20251009-153000`                               |
| `DGXSYSTEM`          | System name (e.g., `MI300X`, `MI325X`, ...)                        | `MI300X`                                        |
| `DATADIR`            | Path to the dataset directory                                      | `/mnt/data/llama2/70b/data`                     |
| `MODELDIR`           | Path to the model checkpoint directory                             | `/mnt/data/llama2/70b/model`                    |
| `IMAGE_NAME`         | Docker image to use                                                | `llmboost/mb-llmboost-training:mlperf-5.1-prod` |
| `NCCL_SOCKET_IFNAME` | Comma-separated list of RDMA-enabled network interfaces            | `ens11np0,ens12np0,...`                         |
| `GLOO_SOCKET_IFNAME` | Comma-separated list of Ethernet interfaces                        | `eno1,eno2`                                     |
| `MASTER_ADDR`        | Hostname or IP of the master node (use node-0 for multi-node)      | `node-0.cluster.local`                          |
| `MASTER_PORT`        | Port for distributed training communication                        | `29500`                                         |
| `WORLD_SIZE`         | Total number of nodes participating                                | `1`, `2`, `4`, ...                              |
| `RANK`               | Rank of the current node (0-based index)                           | `0`, `1`, ...                                   |
| `NCCL_DEBUG`         | NCCL log level (optional, e.g., `INFO`, `WARN`, `VERSION`)         | `INFO`                                          |
| `CONTAINER`          | Name for the Docker container (recommended: unique per run/rank)   | `mb_llmboost_llama2_20251009-153000_0`          |

---

# 3. Running the Benchmark with `docker_run.sh`

This repository provides a robust, production-grade script `docker_run.sh` to automate the setup and execution of MLPerf Training benchmarks in a reproducible manner. This script is intended for both single-node and multi-node deployments and is suitable for use by customers, auditors, and researchers.

## 3.1. Usage

### Single-Node Run

On a single node, set `WORLD_SIZE=1` and `RANK=0`. After exporting all required variables, launch:

```bash
bash docker_run.sh
```

### Multi-Node Run

On each node, set `WORLD_SIZE` to the total number of nodes and `RANK` to the unique index of the node (starting from 0). Ensure all nodes use the same `RUN_DATETIME` and `CONTAINER` naming convention.

**Example for 2 nodes:**

- On node-0:
  ```bash
  export RANK=0
  bash docker_run.sh
  ```
- On node-1:
  ```bash
  export RANK=1
  bash docker_run.sh
  ```

**Note:** All nodes must be able to reach each other over the specified network interfaces.

## 3.2. What the Script Does

- **Validates** all required environment variables.
- **Pulls** the specified Docker image.
- **Stops** any running Docker containers to avoid conflicts.
- **Tunes** system settings for optimal MLPerf performance (requires `sudo`).
- **Launches** a Docker container with the correct environment and mounts.
- **Starts** the MLPerf benchmark inside the container, logging output to `mlperf-logs/$WORLD_SIZE/`.

## 3.3. Output and Logs

- Logs for each run are stored in `mlperf-logs/$WORLD_SIZE/`, named as `${RUN_DATETIME}-rank_${RANK}_of_${WORLD_SIZE}.log`.
- Docker container logs and system output are also available for debugging.

## 3.4. Troubleshooting

- **Missing environment variables:** The script will exit with an error listing any unset or empty variables.
- **Permission errors:** Some system tuning commands require `sudo`. Ensure your user has appropriate permissions.
- **Network issues:** Ensure all nodes can communicate over the specified interfaces and ports.
- **Docker issues:** Make sure Docker is running and your user can run Docker commands.

## 3.5. Customization

- You may modify the script to adjust system tuning, Docker options, or mount additional volumes as needed for your environment.
- For advanced networking or storage setups, update `NCCL_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, and mount points accordingly.

## 3.6. Reproducibility

- Always record the exact environment variable values, Docker image version, and hardware/software configuration for auditability.
- Use the same `RUN_DATETIME` across all nodes in a multi-node run for consistent log aggregation.

---

# 4. Manual/Advanced Benchmark Usage

This section provides a step-by-step guide for advanced users who wish to run the benchmark manually, with explanations for each part of the process and the scripts involved.

## 4.1. System Tuning (`runtime_tunables.sh`)

Before running the benchmark, apply AMD-recommended system tunings for MI300X systems. This script must be run **once per node after each reboot**.

**Command:**
```bash
bash runtime_tunables.sh
```

**What this script does:**
- Drops filesystem caches to ensure clean memory state.
- Loads the AMD GPU kernel module.
- Enumerates available ROCm agents (GPUs).
- Sets CPU idle and frequency governor to performance mode.
- Disables kernel features that may impact performance (NMI watchdog, NUMA balancing, address space randomization).
- Enables transparent hugepages and defragmentation for improved memory performance.

**Script contents:**
```bash
echo 3 | sudo tee /proc/sys/vm/drop_caches
sudo modprobe amdgpu
rocm_agent_enumerator
sudo cpupower idle-set -d 2
sudo cpupower frequency-set -g performance
echo 0 | sudo tee /proc/sys/kernel/nmi_watchdog
echo 0 | sudo tee /proc/sys/kernel/numa_balancing
echo 0 | sudo tee /proc/sys/kernel/randomize_va_space
echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/enabled
echo 'always' | sudo tee /sys/kernel/mm/transparent_hugepage/defrag
```
> **Note:** Most commands require `sudo` privileges.

---

## 4.2. Running the Benchmark Container (`docker_run.sh`)

After system tuning, use `docker_run.sh` to launch the MLPerf benchmark in a Docker container. This script automates environment validation, container setup, and benchmark execution.

**Step-by-step:**

1. **Ensure all required environment variables are set.**
   - See the "Environment Setup" section above for details.

2. **Run the script:**
   ```bash
   bash docker_run.sh
   ```

**What this script does:**

- **Validates environment variables:** Checks that all required variables are set and non-empty.
- **Pulls the Docker image:** Ensures the specified MLPerf container image is available locally.
- **Stops existing containers:** Prevents conflicts by stopping any running Docker containers.
- **Starts the Docker container:** Launches the MLPerf container with all necessary mounts, devices, and environment variables.
- **Verifies environment inside the container:** Confirms that the container is running and environment variables are correctly set.
- **Copies configuration files (if needed):** Ensures the correct config files are available inside the container.
- **Runs the MLPerf benchmark:** Executes the `llmboost mlperf` command inside the container, passing all relevant distributed training parameters.
- **Logs output:** Captures all output to a log file in `mlperf-logs/$WORLD_SIZE/`.

**Relevant script excerpt:**
```bash
# Validate environment variables
# ...existing code...

# Pull Docker image and stop existing containers
docker pull $IMAGE_NAME
docker ps -q | xargs -r docker stop

# Start the container
docker run --rm --init --detach \
    --network host --ipc host --uts host \
    --cap-add SYS_PTRACE \
    --security-opt seccomp=unconfined \
    --group-add video \
    --device /dev/dri:/dev/dri \
    --device /dev/kfd:/dev/kfd \
    -v $DATADIR:/data \
    -v $MODELDIR:/ckpt \
    -w /workspace \
    -e ... # (all required env vars)
    --name "${CONTAINER}" \
    $IMAGE_NAME sleep infinity

# Wait for container to start, verify, and run benchmark
# ...existing code...

docker exec \
    -e NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
    -e GLOO_SOCKET_IFNAME="$GLOO_SOCKET_IFNAME" \
    ... # (other env vars)
    "${CONTAINER}" \
    llmboost mlperf \
      --MASTER_ADDR "${MASTER_ADDR}" \
      --RANK        "${RANK}" \
      --config_sh   conf/config_${DGXSYSTEM}_${WORLD_SIZE}x8x1.sh \
      2>&1 | tee $HOST_LOG_FILE
```

---

## 4.3. Direct Command-Line Usage

For users who wish to bypass the automation scripts and run the benchmark directly, use the following commands:

> Note: In multi-node runs, you must replace `MASTER_ADDR` with the IP address of the main node (typically `node-0`) in your cluster. Also, please modify the `export NCCL_SOCKET_IFNAME=...` inside `config_MI325X_*x8x1.sh` according to your network cards setup.

#### Single Node Benchmark
```bash
llmboost mlperf --config_sh conf/config_MI325X_1x8x1.sh 2>&1 | tee "log_single_node.txt"
```

#### Multi-Node Benchmark (2-nodes)
```bash
# On node-0
llmboost mlperf --MASTER_ADDR <MASTER_NODE_IP> --RANK 0 --config_sh conf/config_MI325X_2x8x1.sh 2>&1 | tee "log_2_nodes.txt"

# On node-1
llmboost mlperf --MASTER_ADDR <MASTER_NODE_IP> --RANK 1 --config_sh conf/config_MI325X_2x8x1.sh
```

---
