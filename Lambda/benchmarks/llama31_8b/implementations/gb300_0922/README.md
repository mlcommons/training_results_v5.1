## Running NVIDIA Large Language Model Llama 3.1 405B PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA Large Language Model Llama 3.1 405B PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 2.5TB disk space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPUs are not required for dataset preparation.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:llama31_405b-pyt .
```

make sure that container is accessible on your Slurm system.

### 3.2 Prepare dataset

Download the dataset to your desired dataset path.

```bash
cd <desired/dataset/path>
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d 405b https://training.mlcommons-storage.org/metadata/mixtral-8x22b-preprocessed-c4-dataset.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d 405b/tokenizer https://training.mlcommons-storage.org/metadata/mixtral-8x22b-tokenizer.uri
```
For more details please refer to the [instructions](https://github.com/mlcommons/training/tree/master/large_language_model_pretraining/nemo#preprocessed-data-download) from the reference to download the dataset and the tokenizer. After following the instructions, you should be able to find the following necessary files under the following environment variables: 

 - Under your dataset path you should have:
   - 405b:
      - `c4-train.en_<number>_text_document` where `number` belongs to 0~7. 
      - `c4-validation-91205-samples`
   - tokenizer:  
      - `special_tokens_map.json`
      - `tokenizer.json`
      - `tokenizer.model`
      - `tokenizer.model.v1`
      - `tokenizer_config.json`

We can remove the unnecessary files by running:

```bash
bash cleanup.sh
```

The final content under 405b should be:

```
c4-train.en_6_text_document.bin
c4-train.en_6_text_document.idx
c4-train.en_7_text_document.bin
c4-train.en_7_text_document.idx
c4-validation-91205-samples.en_text_document.bin
c4-validation-91205-samples.en_text_document.idx
```

### 3.4 Model and checkpoint preparation

#### 3.4.1 Publication/Attribution

[Megatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/intro.html) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository uses [NeMo Megatron](https://github.com/NVIDIA/NeMo). NeMo Megatron GPT has been integrated with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Transformer Engine enables FP8 training on NVIDIA Hopper GPUs.

#### 3.4.2 List of Layers

The model largely follows the [Llama 3.1 405B paper](https://arxiv.org/abs/2407.21783). The only difference is that we replace the paper's TikTokenizer with the Mixtral 8x22b tokenizer in this benchmark. Please refer to the [Model details section](https://github.com/mlcommons/training/tree/master/large_language_model_pretraining/nemo#model-details) from the reference for more details. 

#### 3.4.3 Model checkpoint
In the benchmarking region, we resume training from Meta's official HuggingFace checkpoint. Please refer to the [instructions](https://github.com/mlcommons/training/tree/master/large_language_model_pretraining/nemo#checkpoint-download) from the reference to download the BF16 model checkpoint. 

**NOTE**: Before you proceed, make sure that your current working directory is able to hold >1.5TB of data. 

Assuming that you are running the download command under a given directory, with its location stored under `LOAD_CHECKPOINTS_PATH` environment variable. After the checkpoint is downloaded, you should be able to find a `405b` folder which holds a `context` and `weights` subfolder under the current directory: 

```
<LOAD_CHECKPOINTS_PATH>
└── 405b
    ├── context
    │   ├── nemo_tokenizer
    │   │   ├── special_tokens_map.json
    │   │   ├── tokenizer_config.json
    │   │   └── tokenizer.json
    │   ├── io.json
    │   └── model.yaml
    └── weights
        ├── __0_0.distcp
        ├── __0_1.distcp
        ├── .metadata
        ├── common.pt
        └── metadata.json
```

By default, when we run the container, we will mount `LOAD_CHECKPOINTS_PATH` to `/load_checkpoints` in the container. Thus, you should set `export LOAD_CHECKPOINT="/load_checkpoints/405b"` to ensure that the `405b` folder is accessed in the container. 

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:
```bash
export DATADIR="<desired/dataset/path>"
export LOAD_CHECKPOINTS_PATH="</path/to/your/downloaded/checkpoint>"
export LOAD_CHECKPOINT="</load_checkpoints/405b>"
export LOGDIR=</path/to/output/dir>  # set the place where the output logs will be saved
export CONT=<docker/registry>/mlperf-nvidia:llama31_405b-pyt
source config_GB200_128x4x144xtp4pp8cp2_cg.sh  # select config and source it
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

Replace `/path/to/your/downloaded` prefix with your existing downloaded path in Section 3.

All configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>xtpXppYcpZ.sh`, where X represents tensor parallel, Y represents pipeline parallel, and Z represents context parallel.

# 5. Quality

### Quality metric
Log Perplexity

### Quality target
5.6

### Evaluation frequency
Evaluate after every 46,080 sequences (=377.49B tokens)

### Evaluation thoroughness
Evaluation on the validation subset that consists of 5,760 sequences (=47.19B tokens).


# 6. Additional notes

### Config naming convention

Configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>xtpXppYcpZ.sh`, where X represents tensor parallel (TP), Y represents pipeline parallel (PP), and Z represents context parallel (CP).

Notice here that: 

```
MP = TP * PP * CP
DP = WS // MP = (NNODES * GPUS_PER_NODE) / (TP * PP * CP)
miniBS = GBS // DP
```

where: 
```
MP = model parallelism
TP = tensor parallelism
PP = pipeline parallelism
DP = data parallelism
WS = world size (number of nodes x number of gpus per node)
GBS = global batch size
```

Note: changing `MICRO_BATCH_SIZE` doesn't affect GBS or any of the above parameters.
Effectively it controls gradient accumulation (`GA = miniBS // microBS`).

Recommendation on adjusting the knobs: 
1. GBS should be divisible by `DP * VP`, where VP represents Virtual Pipelining, controlled by environment variable `INTERLEAVED_PIPELINE`. 
2. Model's number of layers, controlled by `OVERWRITTEN_NUM_LAYERS` knob (with default 126), should be divisible by PP * VP. 
   1. It's also recommended that, if you choose to adjust this knob, then you should export `LOAD_CHECKPOINT=""` to disable checkpoint loading, otherwise you will be loading a checkpoint with more layers to a model with fewer layers, which might cause issues. 


### Seeds
NeMo produces dataset index shuffling only on one process and holds the `SEED` value in the file name.
Thus, all processes need to have the same value of `SEED` otherwise will not be able to read the data.
The `SEED` environment variable can be set prior to launching the job, otherwise it is set in `run.sub`.
