## Running NVIDIA HugeCTR DLRM DCNv2 MLPerf Benchmark

This file contains the instructions for running the NVIDIA HugeCTR DLRM DCNv2 MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 7TB of fast storage space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPU is not needed for preprocessing scripts, but is needed for training.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:recommendation-hugectr
```

### 3.2 Prepare the input dataset

Start the container, replacing `</path/to/your/data>` with existing path to where you want to save the dataset:

```bash
docker run -it --rm \
--network=host --ipc=host \
-v </path/to/your/data>:/data \
<docker/registry>/mlperf-nvidia:recommendation-hugectr
```

#### 3.2.1 Run preprocessing steps to get data in NumPy format

```bash
./scripts/process_Criteo_1TB_Click_Logs_dataset.sh \
    /data/criteo_1tb/raw_input_dataset_dir \
    /data/criteo_1tb/temp_intermediate_files_dir \
    /data/criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir
```

As a result, files named: `day_*_labels.npy`, `day_*_dense.npy` and `day_0_sparse.npy` will be created (3 per each of 24 days in the original input dataset, 72 files in total). Once completed, the output data can be verified with md5sums provided in [md5sums_preprocessed_criteo_click_logs_dataset.txt](https://github.com/mlcommons/training/blob/master/recommendation_v2/torchrec_dlrm/md5sums_preprocessed_criteo_click_logs_dataset.txt) file.

#### 3.2.2 Create a synthetic multi-hot Criteo dataset

This step produces multi-hot dataset from the original (one-hot) dataset.

```bash
python scripts/materialize_synthetic_multihot_dataset.py \
    --in_memory_binary_criteo_path /data/criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir \
    --output_path /data/criteo_1tb_sparse_multi_hot \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --multi_hot_sizes 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
    --multi_hot_distribution_type uniform
```

#### 3.2.3 Convert NumPy dataset to raw format

Because HugeCTR uses, among others, [raw format](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#raw) for input data, we need to convert NumPy files created in the preceding steps to this format. To this end, use `preprocessing/convert_to_raw.py` script.

```bash
python preprocessing/convert_to_raw.py \
   --input_dir_labels_and_dense /data/criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir \
   --input_dir_sparse_multihot /data/criteo_1tb_sparse_multi_hot \
   --output_dir /data/criteo_1tb_multihot_raw \
   --stages train val
```

As a result, `train_data.bin` and `val_data.bin` will be created. Once done, the output files can be verified with the md5sums provided in `preprocessing/md5sums_raw_dataset.txt` file.

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

### NVIDIA DGX H100

To launch training with a Slurm cluster, run:

```bash
export DATADIR=/path/to/train/criteo_multihot_raw
export DATADIR_VAL=/path/to/val/criteo_multihot_raw
source config_DGXH100_<config>.sh  # select config
sbatch -N $DGXNNODES run.sub
```
