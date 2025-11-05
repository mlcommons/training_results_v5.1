## Steps to launch training

### eos-dfw_n64_ngc25.01_nemo

Launch configuration and system-specific hyperparameters for the
eos-dfw_n64_ngc25.01_nemo submission are in the
`benchmarks/llama31_405b/implementations/eos-dfw_n64_ngc25.01_nemo/config_DGXH100_64x8x288xtp8pp8cp2.sh` script.

Steps required to launch training for eos-dfw_n64_ngc25.01_nemo.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_DGXH100_64x8x288xtp8pp8cp2.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
