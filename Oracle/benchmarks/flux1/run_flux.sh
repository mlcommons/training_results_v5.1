#!/bin/bash

BASE_DIR=/mnt/lfs/sce/mlperftv51/flux
export DATAROOT="$BASE_DIR/data/energon"  # set your </path/to/dataset>
export LOGDIR="$BASE_DIR/logs"  # set the place where the output logs will be saved
export CONT=$BASE_DIR/flux_20250930.sqsh  # set the container url
#source config_B200_02x08x32.sh  # select config and source it
source config_B200_09x08x32.sh  # select config and source it
sbatch --exclusive --gpus-per-node=8 -N $DGXNNODES -t $WALLTIME run.sub 
