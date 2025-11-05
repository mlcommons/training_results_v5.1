#!/bin/bash

cd ../pytorch
export CONT=./nvdlfwea+mlperftv51+llama2_70b_lora-amd+.sqsh
source config_G894-AD1_1x8x1xtp1pp1cp1.sh
export DATADIR=/path/to/datasets
export MODEL=/path/to/model
export LOGDIR=/path/to/logdir
export MLPERF_SUBMISSION_ORG=GigaComputing
export MLPERF_SUBMISSION_PLATFORM=G894-AD1

sbatch -N $DGXNNODES -t $WALLTIME run.sub
