# Steps to launch training on Nebius Soperator cluster

## Prepare the environment

[Soperator (Slurm on k8s)](https://github.com/nebius/soperator) cluster is provisioned and configured by Nebius engineers. Once the cluster is provisioned and ready to be used, you will receive a login service IP for SSH access.

You will be submitting Slurm jobs from a login node.
```
ubuntu@login-0:/$ sinfo
PARTITION AVAIL  TIMELIMIT  NODES  STATE NODELIST
main*        up   infinite      1   idle worker-[0-127]
```

To use docker, you need to connect to a worker node:

```
ubuntu@login-0:~# ssh worker-0
Last login: Thu May  1 21:47:15 2025 from 10.0.25.112
ubuntu@worker-0:~# docker ps
CONTAINER ID   IMAGE     COMMAND   CREATED   STATUS    PORTS     NAMES
```

## Build docker image and push it to a registry

```bash
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

## Submit the training job

For submitting Slurm jobs, make sure you are on one of the `login` nodes.

To start the training:
```bash
export PMIX_GDS_MODULE=^ds12
export PMIX_MCA_gds=^ds12
export NCCL_NVLS_ENABLE=1

export PREPROC_DATA=</path/to/your/preprocessed_c4>
export SPM=</path/to/your/tokenizer.model>
export LOAD_CHECKPOINTS_PATH=</path/to/your/downloaded/checkpoint>
export LOAD_CHECKPOINT="/load_checkpoints/405b"
export CHECKPOINT_NAME="/load_checkpoints/405b"
export LOGDIR=</path/to/output/dir>  # set the place where the output logs will be saved
export CONT=<docker/registry:benchmark-tag>

source config_DGXH200_64x8x144xtp4pp8cp2.sh
sbatch -N ${DGXNNODES} --gres=gpu:8 --time=${WALLTIME} run.sub
```
