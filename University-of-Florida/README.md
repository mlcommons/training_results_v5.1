# MLPerf v5.1 University of Florida Submission

This is a repository of University of Florida's submission to the MLPerf Training v5.1 benchmark.  It
includes implementations of the benchmark code optimized for running on NVIDIA
GPUs.  The reference implementations can be found elsewhere:
https://github.com/mlcommons/training.git

# v5.1 release

This readme was updated in Sep 2025, for the v5.1 round of MLPerf Training.

# Contents

Each implementation in the `benchmarks` subdirectory provides the following:
 
* Code that implements the model in at least one framework.
* A Dockerfile which can be used to build a container for the benchmark.
* Documentation on the dataset, model, and machine setup.

# Running Benchmarks

These benchmarks have been tested on the following machine configuration:

* NVIDIA DGX B200 SuperPOD system.
* The required software stack includes Slurm and Apptainer for running containers. 

Generally, a benchmark can be run with the following steps:

1. Follow the instructions in the README to download and format the input data and any required checkpoints.
2. Build the containers with Apptainer.
3. Source the appropriate `config_*.sh` file.
4. `sbatch run_*.sub`
