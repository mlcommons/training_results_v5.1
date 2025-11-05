## Running Large Language Model Llama 3.1 8B PyTorch MLPerf Benchmark

This file contains the instructions for running the Large Language Model Llama 3.1 8B PyTorch MLPerf Benchmark on NVIDIA hardware.

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

The current codebase is using the c4/en/3.0.1 dataset from [HuggingFace/AllenAI](https://huggingface.co/datasets/allenai/c4) for train and evaluation. 

### 3.2 Preprocessed data download

The pre-tokenized dataset and the tokenizer are available to download. More instructions to download on Windows are available [here](https://training.mlcommons-storage.org/index.html). You can download using the following commands:

```bash
# data 
# go to the path where you want the data to be downloaded
# use the same path in config when exporting PREPROCESSED_PATH
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d llama3_1_8b_preprocessed_c4_dataset https://training.mlcommons-storage.org/metadata/llama-3-1-8b-preprocessed-c4-dataset.uri

```bash
# tokenizer 
# go to the path where you want the tokenizer to be downloaded
# use the same path in config when exporting TOKENIZER_PATH
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) -d llama3_1_8b_tokenizer https://training.mlcommons-storage.org/metadata/llama-3-1-8b-tokenizer.uri
```

### 3.2 Raw data downloading [Optional]

We use [AllenAI C4](https://huggingface.co/datasets/allenai/c4) dataset for this benchmark. The original zipped **`json.gz`** files can be downloaded by following AllenAI C4's instruction, and you can download our zipped customized validation dataset from the MLCommons S3 bucket by running the following command: 


```bash
export C4_PATH=""

# download the full C4 files, including all raw train and validations
rclone copy mlc-training:mlcommons-training-wg-public/common/datasets/c4/original/en_json/3.0.1 $C4_PATH -P
```
After downloading, run the following command to process them to zip them into `.gz` format before running the data preprocessing. 

```
bash utils/parallel_compress_json_to_gz.sh
```

Run the following commands to merge all 1024 training files into 8 `json.gz` files, all 8 validation files into a single `json.gz` file, as well as generate our customized validation dataset. Each of the `json.gz` files will subsequently be preprocessed into a pair of megatron dataset files (`.bin` and `.idx`) by our preprocess.sh script. 

```bash
export C4_PATH=""
export MERGED_C4_PATH=""
# more information about this knob can be found in consolidate_data.sh
export N_VALIDATION_SAMPLES=91205

bash utils/consolidate_data.sh
```

### 3.3 Tokenizer
We are using the Llama 3.1 8B tokenizer. To download it, you can run the following commands:
```bash
export TOKENIZER_PATH=""
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.1-8B  --local-dir $TOKENIZER_PATH
```

After the data consolidation is done, we can perform preprocessing using the following commands: 

```bash
# pass in the folder path that contains the Llama tokenizer here
# please refer to the tokenizer section above for more details
export TOKENIZER_PATH=""
# pass in the merged file path here
export MERGED_C4_PATH=""
# this path is used for storing the preprocessed .bin and .idx files
export PREPROCESSED_PATH=""

for index in {0..7}; do
    # please specify the right path to nemo
    python3 </path/to/nemo>/nemo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input "${MERGED_C4_PATH}/c4-train.en_${index}.json.gz" \
    --output-prefix "${PREPROCESSED_PATH}/c4-train.en_${index}" \
    --tokenizer-library huggingface --tokenizer-type ${TOKENIZER_PATH} \
    --dataset-impl mmap --workers 128 &
done
    # please specify the right path to nemo
    python3 </path/to/nemo>/nemo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
    --input "${MERGED_C4_PATH}/c4-validation-91205-samples.en.json.gz" \
    --output-prefix "${PREPROCESSED_PATH}/c4-validation-91205-samples.en" \
    --tokenizer-library huggingface --tokenizer-type ${TOKENIZER_PATH} \
    --dataset-impl mmap --workers 128 & 
wait

```

After the download is complete, you should see files with the following naming conventions under `PREPROCESSED_PATH`, ending with both `.idx` and `.bin`: 
- Training partitions: `c4-train.en_<number>_text_document`
- Validation partitions: `c4-validation-91205-samples.en_text_document`

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:
```bash
export PREPROC_DATA="/path/to/your/preprocessed_c4"
export TOKENIZER="/path/to/your/tokenizer.model"
export LOAD_CHECKPOINTS_PATH=""
export LOAD_CHECKPOINT=""
export LOGDIR=</path/to/output/dir>  # set the place where the output logs will be saved
export CONT=<docker/registry>/mlperf-nvidia:large_language_model-pyt
source config_DGXH100_8x8x1xtp1pp1cp1.sh  # select config and source it
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

Replace `/path/to/your` prefix with your existing path.

All configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>xtpXppYcpZ.sh`, where X represents tensor parallel, Y represents pipeline parallel, and Z represents context parallel.

# 5. Quality

### Quality metric
Log Perplexity

### Quality target
3.3

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
