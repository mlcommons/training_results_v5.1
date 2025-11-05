#!/bin/bash

docker run -it -v /data/ctlee/mlperf_training_5.1/llama2-70b_data:/data \
	--net=host --uts=host \
	--ipc=host --device /dev/dri --device /dev/kfd \
	--security-opt=seccomp=unconfined \
	rocm/amd-mlperf:llama2_70b_training_5.1

