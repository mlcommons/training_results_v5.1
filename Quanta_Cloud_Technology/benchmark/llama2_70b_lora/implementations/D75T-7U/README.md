## Steps to launch training

### QuantaGrid D75T-7U

Launch configuration and system-specific hyperparameters for the QuantaGrid D75T-7U
submission are in the `../<implementation>/pytorch_D75T-7U/config_D74H-7U.sh` script.

For training, we use docker to run our container.

Steps required to launch training on QuantaGrid D75T-7U.

1. Build the docker container and push to a docker registry

```
cd ../pytorch_D75T-7U
docker build --pull -t <docker/registry:benchmark-tag> .
```

2. Launch the training
```
export DATA_DIR="<path/to/dataset/and/model>"
export LOGDIR="<path/to/output/dir>"
export CONT="<docker/registry:benchmark-tag>"
source configs/config_D75T-7U.sh  # use appropriate config
export NEXP=10
bash run_with_docker.sh

