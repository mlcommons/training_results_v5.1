## Steps to launch training

### tyche_ngpu8_ngc25.03_hugectr

Launch configuration and system-specific hyperparameters for the
tyche_ngpu8_ngc25.03_hugectr submission are in the
`benchmarks/dlrm_dcnv2/implementations/hugectr/config_GB200_2x4x6912.sh` script.

Steps required to launch training for tyche_ngpu8_ngc25.03_hugectr.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_GB200_2x4x6912.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
