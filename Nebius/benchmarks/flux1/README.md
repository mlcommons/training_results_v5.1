## Running NVIDIA text-to-image FLUX PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA text-to-image FLUX PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 6TB disk space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPUs are not required for dataset preparation.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:flux1-pyt .
```

make sure that container is accessible on your Slurm system.

### 3.2 Download dataset and preprocess

The dataset download/preprocessing scripts are included in the container. To invoke them, you need either a docker or slurm/enroot environment. Start the container, replacing `</path/to/dataset>` with the existing path to where you want to save the dataset and the model weights/tokenizer:

```bash
docker run -it --rm --network=host --ipc=host --volume </path/to/dataset>:/dataset <docker/registry>/mlperf-nvidia:flux1-pyt
# if using srun, instead do: srun --nodes=1 -t 24:00:00 --pty --container-image=<docker/registry>/mlperf-nvidia:flux1-pyt --container-mounts=</path/to/dataset>:/dataset -p <partition> -A <account> /bin/bash 
# now you should be inside the container in the /workspace/flux directory
pip install datasets # install hf_datasets

# download dataset and empty encodings
cd /dataset
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-cc12m-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-coco-preprocessed.uri
bash <(curl -s https://raw.githubusercontent.com/mlcommons/r2-downloader/refs/heads/main/mlc-r2-downloader.sh) https://training.mlcommons-storage.org/metadata/flux-1-empty-encodings.uri

# convert to webdataset format
mkdir energon
python /workspace/flux/scripts/to_webdataset.py --input_path /dataset/cc12m_preprocessed --output_path /dataset/energon/train --num_workers 8
python /workspace/flux/scripts/to_webdataset.py --input_path /dataset/coco_preprocessed --output_path /dataset/energon/val --num_workers 8

# prepare energon metadata
cd energon
energon prepare --split-parts 'train:train/.*' --split-parts 'val:val/.*' ./
# Select y for duplicate keys
# Select y for creadint interactively
# Select class 11

# copy over empty_encodings
cp -r ../empty_encodings .

# (Optional) to reclaim space delete /dataset/cc12m_preprocessed and /dataset/coco_preprocessed
```
or by simply run the download/preprocessing slurm script:

```bash
export DATADIR="</path/to/dataset>"  # set your </path/to/dataset>
export CONT=<docker/registry>/mlperf-nvidia:flux1-pyt
sbatch -N1 -t 24:00:00 download.sub --download --preprocess
```

After, the data structure should look like:
```
/dataset
├── energon
│   ├── train
│   ├── val
│   └── empty_encodings
```

Exit the container.

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:

```bash
export DATAROOT="</path/to/dataset>/energon"  # set your </path/to/dataset>
export LOGDIR="</path/to/output_logdir>"  # set the place where the output logs will be saved
export CONT=<docker/registry>/mlperf-nvidia:flux1-pyt
source config_<system>.sh  # select config and source it
sbatch -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
```

## 5. Evaluation

### Quality metric
Validation loss averaged over 8 equidistant time steps [0, 7/8], as described in [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/pdf/2403.03206).
The validation dataset is prepared in advance so that each sample is associated with a timestep.
This is an integer from 0 to 7 inclusive, and thus should be divided by `8.0` to obtain the timestep.

The algorithm is as follows:

```pseudocode
ALGORITHM: Validation Loss Computation

INPUT:
  - validation_samples: set of validation data samples

INITIALIZE:
  - sum[8]: array of zeros for accumulating losses
  - count[8]: array of zeros for counting samples per timestep

FOR each sample, timestep in validation_samples:
    loss = forward_pass(sample, timestep=t/8)
    sum[t] += loss
    count[t] += 1

mean_per_timestep = sum / count
validation_loss = mean(mean_per_timestep)

RETURN validation_loss
```

As we ensure that the validation set has an equal number of samples per timestep, 
a simple average of all loss values is equivalent to the above.

### Quality target
0.586
### Evaluation frequency
Every 262,144 training samples.
### Evaluation thoroughness
29,696 samples
