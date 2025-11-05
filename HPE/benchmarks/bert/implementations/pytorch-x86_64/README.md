## Running NVIDIA BERT PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA BERT PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 2TB disk space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPUs are required for dataset preparation.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:language_model-pyt
```

### 3.2 Prepare dataset

Start the container, replacing `</path/to/your/data>` with the existing path to where you want to save the BERT data:

```bash
docker run -it --gpus=all --runtime=nvidia --ipc=host -v </path/to/your/data>:/workspace/bert_data <docker/registry>/mlperf-nvidia:language_model-pyt
```

To get the trainset in both standard and packed format (i.e. more than one sequence are packed into trainng sample to minimize padding), run within the container

```bash
cd /workspace/bert
./input_preprocessing/prepare_data.sh --outputdir /workspace/bert_data --packed-data
```

(note: this script requires access to GPU only for `convert_tf_checkpoint.py` script and may take around 24 hours to complete)

This script will download the required data and model files from [MLCommons members Google Drive location](https://drive.google.com/drive/u/0/folders/1oQF4diVHNPCclykwdvQJw8n_VIWwV0PT) creating the following foldes structure:

(Mind that since the v2.1 there is additional step in the data preprocessing that introduces additional shuffling at the samlpes level. If you want to use the `data processed in earlier rounds`, it is strongly suggested that you run lines 123-134 from prepare_data.sh script.)

```
/workspace/bert_data/
├── download
│   ├── bert_reference_results_text_md5.txt
│   ├── results4  [502 entries]
│   ├── results_text.tar.gz
├── hdf5
│   ├── eval
│   │   ├── eval_all.hdf5
│   │   └── part_eval_10k.hdf5
│   ├── eval_varlength
│   │   └── part_eval_10k.hdf5
│   ├── training  [500 entries]
│   └── training-4320
│       ├── hdf5_4320_shards_uncompressed  [4320 entries]
│       ├── hdf5_4320_shards_varlength  [4320 entries]
│       └── hdf5_4320_shards_varlength_shuffled  [8640 entries]
├── packed_data  [8640 entries]
├── per_seqlen  [512 entries]
├── per_seqlen_parts  [500 entries]
└── phase1
    ├── bert_config.json
    ├── model.ckpt-28252.data-00000-of-00001
    ├── model.ckpt-28252.index
    ├── model.ckpt-28252.meta
    ├── model.ckpt-28252.pt
    └── vocab.txt
```

The resulting HDF5 files store the training/evaluation dataset as a variable-length types (https://docs.h5py.org/en/stable/special.html). Note that these do not have a direct Numpy equivalents and require "by-sample" processing approach. The advantage is significant storage requirements reduction.

The `prepare_data.sh` script does the following:
* downloads raw data from GoogleDrive
* converts the training data to hdf5 format (each of 500 data chunks)
* splits the data into appropriate number of shards (for large scale training we recommend using 4320 shards - the default)
* 'compresses' the shards converting fixed-length hdf5 to variable-length hdf5 format
* applies the same procedure to evaluation data
* converts the seed checkpoint from tensorflow 1 to pytorch format

### 3.3 Clean up
To de-clutter `bert_data/` directory, you can remove `download`, `hdf5/training`, and `hdf5/training-4320/hdf5_4320_shards_uncompressed` directories, but if disk space is not a concern, it might be good to keep these to debug any data preprocessing issue.

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:
```bash
export EVALDIR="/path/to/your/data/hdf5/eval_varlength"
export DATADIR_PHASE2="/path/to/your/data/hdf5/training-4320/hdf5_4320_shards_varlength_shuffled"
export DATADIR_PHASE2_PACKED="/path/to/your/data/packed_data"
export CHECKPOINTDIR_PHASE1="/path/to/your/data/phase1"
export LOGDIR=</path/to/output/dir> # set the place where the output logs will be saved
export CONT=<docker/registry>/mlperf-nvidia:language_model-pyt
export CONTAINER_PRELOAD_LUSTRE=0
source config_DGXH100_1x8x48x1_pack.sh  # select config and source it
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub  # you may be required to set --account and --partition here
```

Replace `/path/to/your/data` with your existing path.

All configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>x<GRADIENT_ACCUMULATION_STEPS>.sh`.