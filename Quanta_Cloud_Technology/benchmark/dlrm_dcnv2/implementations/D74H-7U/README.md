## Steps to launch training

### QuantaGrid D74H-7U

Launch configuration and system-specific hyperparameters for the QuantaGrid D74H-7U
submission are in the `../<implementation>/hugectr/config_D74H-7U.sh` script.

Steps required to launch training on QuantaGrid D74H-7U.

1. Build the docker container and push to a docker registry

```
cd ../hugectr
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Transfer the docker image to enroot container image

```
enroot import -o <path_to_enroot_image_name>.sqsh dockerd://<docker/registry:benchmark-tag>
```

3. Launch the training
```
export DATADIR=<path/to/data>
export DATADIR_VAL=<path/to/val_data>
export CONT="<path_to_enroot_image_name>.sqsh"
source config_D74H-7U.sh 
NEXP=10 sbatch --gpus=${DGXNGPU} -N ${DGXNNODES} run.sub

