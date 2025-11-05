# Running AMD LLama2-70B LoRA PyTorch MLPerf Benchmark
This benchmark represents a LLama2-70B LoRA finetuning on the [GovReport](https://gov-report-data.github.io/) dataset.

# 1. Setup Docker Image

## Option 1: Pull Docker Image

```bash
docker pull rocm/amd-mlperf:llama2_70b_training_5.1
```
## Option 2: Build Docker Image

Run the following build command from the root of the repository. The build process will take a while to complete. Ensure that all scripts have write access by running `sudo chmod -R 777'.

```bash
docker build -t rocm/amd-mlperf:llama2_70b_training_5.1 .
```
# 2. Prepare Dataset

## General Information

GovReport is a dataset for long document summarization that consists of reports written by government research agencies. The dataset hosted on the MLPerf drive is already tokenized and packed so that each sequence has length 8192.

The used model is the LLama2-70B with fused QKV. You will need 270GB to download and convert the model.

Note: If you’ve already downloaded the model and dataset, you can skip the section below and proceed directly to the Training section.

## Download and Preprocess Data & Model
To download the model from Huggingface, you'll need to sign the LLAMA 2 COMMUNITY LICENSE AGREEMENT as well as obtain a Hugginface Token `HF_TOKEN`.

Start the docker container by mounting the volume you want to use for downloading the data under `/data` within the container. In this example we use `/data/mlperf_llama2` as the host download directory:

```bash
docker run -it -v /data/mlperf_llama2:/data \
    --net=host --uts=host \
    --ipc=host --device /dev/dri --device /dev/kfd \
    --security-opt=seccomp=unconfined \
    rocm/amd-mlperf:llama2_70b_training_5.1
```

Start the script for downloading and preprocessing data from within the container:

```bash
export HF_TOKEN=<>
bash ./scripts/prepare_data_and_model.sh
```
## Verify Data

The data and model files are stored under `/data` within the container.  
After preprocessing, you should see the following files in the `/data/model` directory:
```
<hash>_tokenizer.model  llama2-70b.nemo  model_config.yaml  model_weights
```
And the following files in the `/data/data` directory:
```
train.npy  validation.npy
```
## Exit Container 

Exit the container by running the below command

```bash
exit
```
# 3. Run Training

### Set Environment

Set the directory for the data, model and results. Ensure that `$LOGDIR` has write access for the results to be written by running `sudo chmod -R 777 $LOGDIR`, In this example we use `/data/mlperf_llama2/results` as the results directory, so please make sure to create this directory.

```bash
export DATADIR=/data/mlperf_llama2
export LOGDIR=/data/mlperf_llama2/results
export CONT=rocm/amd-mlperf:llama2_70b_training_5.1         
```

### Set Configuration

Set appropriate configuration and system-specific hyperparameters:  
MI300x submission configurations are in `config_MI300x_*.sh`  
MI325x submission configurations are in `config_MI325x_*.sh`
MI350x submission configurations are in `config_MI350x_*.sh`
MI355x submission configurations are in `config_MI355x_*.sh`

```bash
source config_MI355X_1x8x1.sh  # use appropriate config
```

### Launch 1 Training Run
If you want to perform a single run, use:
```bash
export NEXP=1
bash run_with_docker.sh
```

### Launch 10 Training Run [Optional]
If you would like to prepare for 10 run submisision, use:

```bash
export NEXP=10
bash run_with_docker.sh
```
After completion, the logs will be available under the directory `$LOGDIR`.

Note:To optimize the machine's performance, the training script will also execute `runtime_tunables.sh` script before any training run.


# 4. Check Quality
### Quality Metric
Cross entropy loss
### Quality Target
0.925
