## Running NVIDIA DGL GNN GATConv MLPerf Benchmark

This file contains the instructions for running the NVIDIA DGL GNN GATConv MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- **2TB CPU memory is required**. 
- At least 6TB disk space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPU is not needed for preprocessing scripts, but is needed for training.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:graph_neural_network-dgl
```

### 3.2 Download and preprocess dataset

Start the container, replacing `<igbh_dataset>`, `<data_path>` and `<graph_path>` with existing paths:
- `<igbh_dataset>`: this path holds all original IGBH-Full data.
- `<data_path>`: this path holds all preprocessed IGBH-Full data, including features and graph structures.
- `<graph_path>`: this path holds only small files, such as graph structures.

```bash
docker run -it --rm --network=host --ipc=host \
-v <igbh_dataset>:/data \
-v <data_path>:/converted \
-v <graph_path>:/graph \
<docker/registry>/mlperf-nvidia:graph_neural_network-dgl
```

#### 3.2.1 Download dataset

To make it more convenient, we have copied the reference's dataset downloading script as-is from the reference branch to docker image. To download the dataset run inside the container:

```bash
bash utility/download_igbh_full.sh
```
Please notice that the downloading takes about 2-3 days to complete.

#### 3.2.2 Generate seeds
To generate the seeds for training and validation run inside the container: 

```bash
python3 utility/split_seeds.py --dataset_size="full" --path /data/igbh
```

After we have downloaded the dataset and prepared seeds, the original FP32-precision IGBH-Full data should reside under path `/data/igbh`. The directory should look like this: 

```
/data
│
└───igbh
    │
    └───full
        │
        └───processed
            │
            └───train_idx.pt
            └───val_idx.pt
            └───paper
            |   │
            |   └───node_feat.npy
            |       node_label_2K.npy
            |       paper_id_index_mapping.npy
            |       ...
            │
            └───...
```

After training seed generation is complete, we can run the following command to verify the correctness of the generated training seeds: 

```bash
python3 utility/verify_dataset_correctness.py --data_dir /data/igbh --dataset_size full
```

#### 3.2.3 Preprocess dataset
Our preprocessing script is set up so that we copy everything to a new directory and later we can directly train from that directory, not touching the original downloaded dataset at all. **This preprocessing step does NOT require GPU**. 

To perform FP16 conversion and graph format conversion run inside the container: 
```bash
python3 utility/preprocessing.py \
--data_dir /data/igbh
--convert_dir /converted
--precision float16 `# we can specify float32 for FP32 features, but it is not recommended` \
--size full \
--shuffle paper author conference journal fos institute \
--graph_storage_copy /graph \
--seed 0 \
--concat_features
```

After this script is done, you should see files under a folder named `/converted/full/float16` directory and `/graph/full` directory inside the container. Most importantly, a `config.yml` file should exist under `/converted/full/float16`, which is crucial for subsequent data loading. The directories should look like this:

```
/converted
└── full
    └── float16 (contains 14 files, size 2.1T)
        └── config.yml
        └── *_node_feat.bin
        └── *_shuffle_map.pt

/graph
└── full (contains 37 files, size 194G)
    └── node_label_*.npy
    └── *_idx.pt
    └── ...
```

The above script should take about 3hr 50min to finish, measured on a single DGX H100 node. 

#### 3.2.4 Optional: FP8 conversion

We have created a separate script to convert the dataset further from FP16 to FP8, using NVIDIA's TransformerEngine. Please notice that **this conversion script requires 1 GPU with 80GB GPU Memory**. If GPU is present in the docker host system, we can launch a container with GPU by adding **`--gpus all`** argument to the container launch command: 

```bash
docker run -it --rm --network=host --ipc=host --gpus all \
-v <igbh_dataset>:/data \
-v <data_path>:/converted \
-v <graph_path>:/graph \
<docker/registry>/mlperf-nvidia:graph_neural_network-dgl
```

To further perform such conversion, we run the following command inside the container with GPU mounted: 

```bash
python3 utility/fp8_conversion.py \
--data_dir /converted/full \
--fp8_format e4m3 \
--scale 1.0
```
This should take about 1.5 hours.

After this script is done, `/converted/full` should contain two directories `float16` and `float8` - with FP16 features and FP8 features respectively. Now we can subsequently enable training with FP8 features using flag `FP8_EMBEDDING=1`.

**Note**: Please notice that FP8 features are set as the default option. To use FP16 features, we need to explicitly additionally set `FP8_EMBEDDING=0` environment variable when launching runs. 

#### 3.2.5 Optional: Patching training seeds

If the training seeds are incorrectly generated by any means (incorrect command line arguments, incorrect random seeds, etc), and needs to be patched, we do not need to run the full preprocessing workflow again to patch them, and can instead follow the following procedure. 

Assuming that we're still in the same container, with: 

- Original IGBH dataset path mounted on `/data`
- Preprocessed features mounted under `/converted`
- Preprocessed Graph copy mounted under `/graph`

Then, we should see `train_idx.pt` and `val_idx.pt` under the following paths: 

- `/data/igbh/full/processed`
- `/converted/full`
- `/graph/full`

Inside this container, we run again the `split_seeds.py`. Attaching an example command with **all correct arguments**: 

```bash
python3 utility/split_seeds.py \
--dataset_size full \
--path /data/igbh \
--validation_frac 0.005 \
--num_classes 2983 \
--random_seed 42
```

Once this is done, we can see that the `train_idx.pt` and `val_idx.pt` files are updated under container path `/data/igbh/full/processed`. Either run through the preprocessing step again to regenerate the final `train_idx.pt` and `val_idx.pt` or utilize `reshuffle_indices.py` to directly shuffle the indices based on the shuffle map generated previously in the preprocessing step.

```bash
python3 reshuffle_indices.py \
--path_to_indices /data/igbh/full/processed/val_idx.pt \
--path_to_shuffle_map /converted/full/paper_shuffle_map.pt \
--path_to_output_indices val_idx_shuffled.pt
```

Then copy `val_idx_shuffled.pt` to `/converted` and `/graph` and rename it to `val_idx.pt`.

```bash
python3 reshuffle_indices.py \
--path_to_indices /data/igbh/full/processed/train_idx.pt \
--path_to_shuffle_map /converted/full/paper_shuffle_map.pt \
--path_to_output_indices train_idx_shuffled.pt
```

Then copy `train_idx_shuffled.pt` to `/converted` and `/graph` and rename it to `train_idx.pt`.

Now you have all files required to launch the training.

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

To launch training with a Slurm cluster, run:

```bash
export DATA_DIR=<data_path>/full  # set the correct <data_path>
export GRAPH_DIR=<graph_path>/full  # set the correct <graph_path>
export LOGDIR=</path/to/your/logdir>  # set the place where the output logs will be saved
export CONT=<docker/registry>/mlperf-nvidia:graph_neural_network-dgl
export FP8_EMBEDDING=1  # use FP8 features
source config_<system>.sh  # select config and source it
sbatch -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
```

Note that this benchmark requires a lot of CPU memory usage on a single node. To achieve optimal performance, 2TB CPU memory is required.
