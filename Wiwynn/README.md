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
```

### 3.2 Download dataset and model + preprocessing

This benchmark uses the [GovReport](https://gov-report-data.github.io/) dataset.

Start the container, replacing `</path/to/dataset>` with the existing path to where you want to save the dataset and the model weights/tokenizer:

```bash
docker run -it --rm --gpus all --network=host --ipc=host --volume </path/to/dataset>:/data <docker/registry>/mlperf-nvidia:llama2_70b_lora-pyt
# now you should be inside the container in the /workspace/ft-llm directory
python scripts/download_dataset.py --data_dir /data/gov_report  # download and preprocess dataset; takes less than 1 minute
python scripts/download_model.py --model_dir /data/model  # download preprocessed model checkpoint in NeMo format used for initialization; could take up to 30 minutes
```

You can also use the `scripts/convert_model.py` script, which downloads the original LLaMA2-70B model and converts it to the NeMo format, e.g. `python scripts/convert_model.py --output_path /data/model`. This script requires that you either: have set the `HF_TOKEN` with granted access to the `meta-llama/Llama-2-70B-hf`, or have already downloaded the LLaMA2-70B checkpoint and set `HF_HOME` to its location.

After both scripts finish you should see the following files in the `/data` directory:

```
/data
├── gov_report
│   ├── train.npy
│   └── validation.npy
└── model
    ├── context
    │   ├── io.json
    │   ├── model.yaml
    │   └── nemo_tokenizer
    └── weights
        ├── common.pt
        ├── metadata.json
        ├── module.decoder.final_layernorm._extra_state
        ├── module.decoder.final_layernorm.weight
        ├── module.decoder.layers.mlp.linear_fc1._extra_state
        ├── module.decoder.layers.mlp.linear_fc1.layer_norm_weight
        ├── module.decoder.layers.mlp.linear_fc1.weight
        ├── module.decoder.layers.mlp.linear_fc2._extra_state
        ├── module.decoder.layers.mlp.linear_fc2.weight
        ├── module.decoder.layers.self_attention.core_attention._extra_state
        ├── module.decoder.layers.self_attention.linear_proj._extra_state
        ├── module.decoder.layers.self_attention.linear_proj.weight
        ├── module.decoder.layers.self_attention.linear_qkv._extra_state
        ├── module.decoder.layers.self_attention.linear_qkv.layer_norm_weight
        ├── module.decoder.layers.self_attention.linear_qkv.weight
        ├── module.embedding.word_embeddings.weight
        └── module.output_layer.weight
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
