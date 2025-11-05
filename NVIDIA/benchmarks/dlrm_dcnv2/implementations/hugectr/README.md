## Running NVIDIA HugeCTR DLRM DCNv2 MLPerf Benchmark

This file contains the instructions for running the NVIDIA HugeCTR DLRM DCNv2 MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 8TB of fast storage space is required.
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

### 3.2 Download dataset

Download dataset from: [https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

Save the downloaded files to an empty directory named "criteo_1tb_raw_input_dataset_dir".

Start the container, replacing `</path/to/your/data>` with the existing path where `criteo_1tb_raw_input_dataset_dir` is stored:

```bash
docker run -it --rm --network=host --ipc=host \
-v </path/to/your/data>:/data \
<docker/registry>/mlperf-nvidia:recommendation-hugectr
```

Unpack `.gz` files:

```bash
cd /data/criteo_1tb_raw_input_dataset_dir
for i in {0..23}; do
    echo "day_${i}.gz"
    gunzip -dk day_${i}.gz
done
```

(Optional) After unpacking, remove `.gz` files.

#### 3.2.1 Install torchrec for preprocessing

```bash
pip install torchrec==1.0.0
```

#### 3.2.2 Run preprocessing steps to get data in NumPy format

```bash
cd /worksapce/dlrm
./scripts/process_Criteo_1TB_Click_Logs_dataset.sh \
    /data/criteo_1tb_raw_input_dataset_dir \
    /data/criteo_1tb_temp_intermediate_files_dir \
    /data/criteo_1tb_numpy_contiguous_shuffled_output_dataset_dir
```

The above script requires 700GB of RAM and takes 3-5 days to complete.

As a result, files named: `day_*_labels.npy`, `day_*_dense.npy` and `day_0_sparse.npy` will be created (3 per each of 24 days in the `/data/criteo_1tb_numpy_contiguous_shuffled_output_dataset_dir` directory, 72 files in total). Once completed, the output data can be verified with:

```bash
./scripts/verify_md5sums_npy_files.sh
```

(Optional) After verifying MD5 hashes, remove raw and temp directories:

```bash
rm -r /data/criteo_1tb_raw_input_dataset_dir
rm -r /data/criteo_1tb_temp_intermediate_files_dir
```

#### 3.2.3 Create a synthetic multi-hot Criteo dataset

This step produces multi-hot dataset from the original (one-hot) dataset.

```bash
python scripts/materialize_synthetic_multihot_dataset.py \
    --in_memory_binary_criteo_path /data/criteo_1tb_numpy_contiguous_shuffled_output_dataset_dir \
    --output_path /data/criteo_1tb_sparse_multi_hot \
    --num_embeddings_per_feature 40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36 \
    --multi_hot_sizes 3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1 \
    --multi_hot_distribution_type uniform
```

The above script takes less than 2 hours to complete.

#### 3.2.4 Convert NumPy dataset to raw format

Because HugeCTR uses, among others, [raw format](https://nvidia-merlin.github.io/HugeCTR/main/api/python_interface.html#raw) for input data, we need to convert NumPy files created in the preceding steps to this format. To this end, use `scripts/convert_to_raw.py` script.

```bash
python scripts/convert_to_raw.py \
   --input_dir_labels_and_dense /data/criteo_1tb_numpy_contiguous_shuffled_output_dataset_dir \
   --input_dir_sparse_multihot /data/criteo_1tb_sparse_multi_hot \
   --output_dir /data/criteo_1tb_multihot_raw \
   --stages train val
```

The above script takes less than 5 hours to complete.

As a result, `train_data.bin` and `val_data.bin` will be created. Once done, the output files can be verified with:

```bash
md5sum /data/criteo_1tb_multihot_raw/train_data.bin  # 4d48daf07cc244f6fa933b832d7fe5a3
md5sum /data/criteo_1tb_multihot_raw/val_data.bin    # c7ca591ad3fd2b09b75d99fa4fc210e2
```

(Optional) After verifying MD5 hashes, remove unneeded files:

```bash
rm -r /data/criteo_1tb_numpy_contiguous_shuffled_output_dataset_dir
rm -r /data/criteo_1tb_sparse_multi_hot
```

Final dataset structure:

```
/data/criteo_1tb_multihot_raw
├── train_data.bin
└── val_data.bin
```

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

To launch training with a Slurm cluster, run:

```bash
export DATADIR=/path/to/data/criteo_1tb_multihot_raw
export DATADIR_VAL=/path/to/data/criteo_1tb_multihot_raw
export LOGDIR="</path/to/output_logdir>"  # set the place where the output logs will be saved
export CONT=<docker/registry>/mlperf-nvidia:recommendation-hugectr
source config_<system>.sh  # select config and source it
sbatch -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
```
