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
docker build -t <docker/registry>/mlperf-nvidia:large_language_model-pyt
```

### 3.2 Prepare dataset

Please refer to the [instructions](https://github.com/mlcommons/training/tree/master/large_language_model_pretraining/nemo#preprocessed-data-download) from the reference to download the dataset and the tokenizer. After following the instructions, you should be able to find the following necessary files under the following environment variables: 

 - Environment variable `PREPROCESSED_PATH`: this environment variable points to the preprocessed dataset. Downloaded files should end with `.idx` and `.bin`
    - `c4-train.en_<number>_text_document` where `number` belongs to 0~7. 
    - `c4-validation-91205-samples`
 - Environment variable `TOKENIZER_PATH`: this environment variable points to the tokenizer used in this benchmark. Downloaded files include: 
   - `special_tokens_map.json`
   - `tokenizer.json`
   - `tokenizer.model`
   - `tokenizer.model.v1`
   - `tokenizer_config.json`

Download cleanup - navigate to `PREPROCESSED_PATH` directory and run:

```bash
rm c4-train.en_0_text_document.bin
rm c4-train.en_0_text_document.idx
rm c4-train.en_1_text_document.bin
rm c4-train.en_1_text_document.idx
rm c4-train.en_2_text_document.bin
rm c4-train.en_2_text_document.idx
rm c4-train.en_3_text_document.bin
rm c4-train.en_3_text_document.idx
rm c4-train.en_4_text_document.bin
rm c4-train.en_4_text_document.idx
rm c4-train.en_5_text_document.bin
rm c4-train.en_5_text_document.idx
mv c4-validation-91205-samples.en_text_document.bin/c4-validationn-91205-samples.en_text_document.bin _c4-validationn-91205-samples.en_text_document.bin
mv c4-validation-91205-samples.en_text_document.idx/c4-validationn-91205-samples.en_text_document.idx _c4-validationn-91205-samples.en_text_document.idx
rm -r c4-validation-91205-samples.en_text_document.bin
rm -r c4-validation-91205-samples.en_text_document.idx
mv _c4-validationn-91205-samples.en_text_document.bin c4-validation-91205-samples.en_text_document.bin
mv _c4-validationn-91205-samples.en_text_document.idx c4-validation-91205-samples.en_text_document.idx
rm c4-validation-small.en_text_document.bin
rm c4-validation-small.en_text_document.idx
rm c4-validation.en_text_document.bin
rm c4-validation.en_text_document.idx
```

The final `PREPROCESSED_PATH` content should be:

```
c4-train.en_6_text_document.bin
c4-train.en_6_text_document.idx
c4-train.en_7_text_document.bin
c4-train.en_7_text_document.idx
c4-validation-91205-samples.en_text_document.bin
c4-validation-91205-samples.en_text_document.idx
```

To use them in our training, we should set `PREPROCESSED_PATH` directory value to the `PREPROC_DATA` variable, and set `TOKENIZER_PATH` value to the `TOKENIZER` variable. In other words: 

```bash
export PREPROC_DATA=$PREPROCESSED_PATH
export TOKENIZER=$TOKENIZER_PATH
```

### 3.3 Model and checkpoint prepration

#### 3.3.1 Publication/Attribution

[Megatron](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/nlp/nemo_megatron/intro.html) is a large, powerful transformer developed by the Applied Deep Learning Research team at NVIDIA. This repository uses [Nemo Megatron](https://github.com/NVIDIA/NeMo). NeMo Megatron GPT has been integrated with [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine). Transformer Engine enables FP8 training on NVIDIA Hopper GPUs.

#### 3.3.2 List of Layers

The model largely follows the [Llama 3.1 405B paper](https://arxiv.org/abs/2407.21783). The only difference is that we replace the paper's TikTokenizer with the Mixtral 8x22b tokenizer in this benchmark. Please refer to the [Model details section](https://github.com/mlcommons/training/tree/master/large_language_model_pretraining/nemo#model-details) from the reference for more details. 

#### 3.3.3 Model checkpoint
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
export PREPROC_DATA="/path/to/your/preprocessed_c4"
export SPM="/path/to/your/tokenizer.model"
export LOAD_CHECKPOINTS_PATH="/path/to/your/downloaded/checkpoint"
export CHECKPOINT_NAME="/load_checkpoints/405b"
export LOGDIR=</path/to/output/dir>  # set the place where the output logs will be saved
export CONT=<docker/registry>/mlperf-nvidia:large_language_model-pyt
source config_DGXH100_72x8x36xtp8pp9cp2.sh  # select config and source it
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

Replace `/path/to/your` prefix with your existing path.

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
