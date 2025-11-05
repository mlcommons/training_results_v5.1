cd /home/hpebench/MLPerf-Data-Shruti/training_results_5.0/HPE/benchmarks/bert/implementations/pytorch

# export MLPERF_RULESET=5.0.0
export MLPERF_SUBMISSION_ORG=HPE
export SUBMISSION_ORG=HPE #which one to use?
export MLPERF_SUBMITTER=HPE
export MLPERF_SYSTEM_NAME="HPE Cray XD685"
export MLPERF_STATUS="Available on-premise"
export MLPERF_DIVISION=open
export MLPERF_NUM_NODES=1 #2
export MLPERF_SCALE=1
export MLPERF_CLUSTER_NAME="HPE Cray XD685"


source config_XD685_1x8x48x1_pack.sh && \
        NEXP=15 \
        CONT=/home/hpebench/MLPerf-Data-Shruti/mlperf5.0-bert-b200-dockerd.sqsh \
        DATADIR_PHASE2=/home/hpebench/MLPerf-Data-Shruti/bert/bert-packed/packed_data \
        CHECKPOINTDIR_PHASE1=/home/hpebench/MLPerf-Data-Shruti/bert/phase1 \
        EVALDIR=/home/hpebench/MLPerf-Data-Shruti/bert/hdf5/eval_varlength \
        WORK_DIR=$PWD \
        LOGDIR=/home/hpebench/MLPerf-Data-Shruti/MLPerf-5.0-Training-Results/bert-logs/1-node-local-final \
        sbatch -N 1 -n 8  --job-name=MLPerf5.0-Training  -w gpu02 $PWD/run.sub

