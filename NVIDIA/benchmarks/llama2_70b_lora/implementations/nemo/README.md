## Running NVIDIA NeMo LLama2-70B LoRA PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA NeMo LLama2-70B LoRA PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 300GB disk space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPU is not needed for preprocessing scripts, but is needed for training.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt .
# optionally: docker push <docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
export CONT=<docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
```

make sure that container is accessible on your Slurm system.

### 3.2 Download dataset and model + preprocessing

This benchmark uses the [GovReport](https://gov-report-data.github.io/) dataset.

The dataset download/preprocessing scripts are included in the container. To invoke them, you need either a docker or slurm/enroot environment. Start the container, replacing `</path/to/dataset>` with the existing path to where you want to save the dataset and the model weights/tokenizer:

```bash
docker run -it --rm --network=host --ipc=host --volume </path/to/dataset>:/data $CONT

# now you should be inside the container in the /workspace/ft-llm directory
python scripts/download_dataset.py --data_dir /data/gov_report  # download and preprocess dataset; takes less than 1 minute
python scripts/download_model.py --model_dir /data/model  # download and preprocess model checkpoint used for initialization; could take up to 30 minutes
```

or simply run the download/preprocessing slurm script:

```bash
export DATADIR="</path/to/dataset>/gov_report"  # set your </path/to/dataset>
export MODEL="</path/to/dataset>/model"  # set your </path/to/dataset>
export CONT=$CONT
sbatch -N1 -t 4:00:00 scripts/download.sub
```

You can also use the `scripts/convert_model.py` script, which downloads the original LLama2-70B model and converts it to the NeMo format, e.g. `python scripts/convert_model.py --output_path /data/model`. This script requires that you either: have set the `HF_TOKEN` with granted access to the `meta-llama/Llama-2-70B-hf`, or have already downloaded the LLama2-70B checkpoint and set `HF_HOME` to its location.

After both scripts finish you should see the following files in the `/data` directory:

```
$tree data/
data/
|-- gov_report
|   |-- train.npy
|   `-- validation.npy
`-- model
    |-- context
    |   |-- artifacts
    |   |   `-- generation_config.json
    |   |-- io.json
    |   |-- model.yaml
    |   `-- nemo_tokenizer
    |       |-- special_tokens_map.json
    |       |-- tokenizer.json
    |       |-- tokenizer.model
    |       `-- tokenizer_config.json
    `-- weights
        |-- __0_0.distcp
        |-- __0_1.distcp
        |-- common.pt
        `-- metadata.json

```

Exit the container.


## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:

```bash
export DATADIR="</path/to/dataset>/gov_report"  # set your </path/to/dataset>
export MODEL="</path/to/dataset>/model"  # set your </path/to/dataset>
export LOGDIR="</path/to/output_logdir>"  # set the place where the output logs will be saved
export CONT=<docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
source config_<system>.sh  # select config and source it
sbatch -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
```

## 5. Evaluation

### Quality metric
Cross entropy loss

### Quality target
0.925

### Evaluation frequency
Every 384 sequences, CEIL(384 / global_batch_size) steps if 384 is not divisible by GBS. Skipping first FLOOR(0.125*global_batch_size+2) evaluations

### Evaluation thoroughness
Evaluation on the validation subset that consists of 173 examples
