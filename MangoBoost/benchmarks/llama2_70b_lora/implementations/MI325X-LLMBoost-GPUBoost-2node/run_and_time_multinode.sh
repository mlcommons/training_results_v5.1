#!/bin/bash

declare -a CMD
readonly node_rank="${NODE_RANK}"
CMD=( ${NSYSCMD} 'torchrun' "--nproc_per_node=${DGXNGPU}" "--nnodes=${DGXNNODES}" "--node_rank=${node_rank}" "--master_addr=${MASTER_ADDR}" "--master_port=${MASTER_PORT}")

# start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"

${CMD[@]} src/train.py
ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

# end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo "ENDING TIMING RUN AT $end_fmt"
# report result
result=$(( end - start ))
result_name="LLM_FINETUNING"
echo "RESULT,$result_name,,$result,AMD,$start_fmt"

exit 0