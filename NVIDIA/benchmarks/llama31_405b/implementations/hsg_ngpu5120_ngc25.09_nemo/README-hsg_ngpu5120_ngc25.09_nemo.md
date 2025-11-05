## Steps to launch training

### hsg_ngpu5120_ngc25.09_nemo

Launch configuration and system-specific hyperparameters for the
hsg_ngpu5120_ngc25.09_nemo submission are in the
`benchmarks/llama31_405b/implementations/hsg_ngpu5120_ngc25.09_nemo/config_GB200_1280x4x16xtp4pp8cp2_cg_fp4.sh` script.

Steps required to launch training for hsg_ngpu5120_ngc25.09_nemo.  The sbatch
script assumes a cluster running Slurm with the Pyxis containerization plugin.

1. Build the docker container and push to a docker registry

```
docker build --pull -t <docker/registry:benchmark-tag> .
docker push <docker/registry:benchmark-tag>
```

2. Launch the training
```
source config_GB200_1280x4x16xtp4pp8cp2_cg_fp4.sh
CONT=<docker/registry:benchmark-tag> DATADIR=<path/to/data/dir> LOGDIR=<path/to/output/dir> sbatch -N ${DGXNNODES} -t ${WALLTIME} run.sub
```
