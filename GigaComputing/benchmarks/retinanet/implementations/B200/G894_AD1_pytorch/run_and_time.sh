#!/bin/bash

cd ../pytorch
export CONT=./nvdlfwea+mlperftv51+retinanet-amd+.sqsh
source config_G894-AD1_001x08x032.sh
export DATADIR=/path/to/datadir
export BACKBONE_DIR=/path/to/backbone_dir
export LOGDIR=/path/to/logdir
export MLPERF_SUBMISSION_ORG=GigaComputing
export MLPERF_SUBMISSION_PLATFORM=G894-AD1

sbatch -N $DGXNNODES -t $WALLTIME run.sub
