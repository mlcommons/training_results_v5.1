# Running AMD LLama3.1-8B Pretraining PyTorch MLPerf Benchmark

Small Language Model pretraining - Llama 3.1 8B

# 1. Setup Docker Image

## Build Docker Image

Run the following build command from the root of the repository. The build process will take a while to complete. Ensure that all scripts have write access by running `sudo chmod -R 777`.

```bash
docker build -t rocm/amd-mlperf:llama31_8b_training_5.1 .
```
# 2. Prepare Dataset and Model

If you have already downloaded the model and dataset then please skip this step.
The current codebase is using the c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4) for train and evaluation.

## Download Preprocessed Data

The pre-tokenized dataset and the tokenizer are available for download.You can navigate in the terminal to your desired download directory and run the following commands to download the dataset and tokenizer. In this example, we're using `/data/mlperf_llama31_8b` as the host download directory.

```bash
# desired download directory
cd /data/mlperf_llama31_8b

# data
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d data https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

# tokenizer
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d model https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri
```

After the download is complete, you should see files with the following naming conventions under the data directory, ending with both `.idx` and `.bin`: 
- Training partitions: `c4-train.en_6_text_document`
- Validation partitions: `c4-validation-91205-samples.en_text_document`

The data directory is ~80 GB and model directory is ~30 GB.

# 3. Run Training

### Set Environment

Set the directory for the data, model and results. Ensure that `$LOGDIR` has write access for the results to be written by running `sudo chmod -R 777 $LOGDIR`, In this example we use `/data/mlperf_llama31_8b/results` as the results directory, so please make sure to create this directory.

```bash
export DATADIR=/data/mlperf_llama31_8b/data
export MODEL=/data/mlperf_llama31_8b/model/
export LOGDIR=/data/mlperf_llama31_8b/results
export CONT=rocm/amd-mlperf:llama31_8b_training_5.1
```

### Set Configuration

Set appropriate configuration and system-specific hyperparameters:\
MI350x submission configurations are in `config_MI350X_*.sh`\
MI355x submission configurations are in `config_MI355X_*.sh`

```bash
source config_MI355X_1x8x1_8b.sh  # use appropriate config
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
### Quality metric

Validation loss

### Quality target

Validation log perplexity = 3.3

### Evaluation frequency

We perform evaluation every **12288** sequences. 

### Evaluation thoroughness

We evaluate using **1024** sequences from our customized validation dataset. 
