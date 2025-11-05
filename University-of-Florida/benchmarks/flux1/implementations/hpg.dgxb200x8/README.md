Please refer to the implementation directory [b200_nemo](../b200_nemo) for more details about data and model preparation.

Copy these files to implementation directory. Modify `flux_training_scale.sh` to set the correct paths for NCCL (from implementation directory), dataset, logs, and container.

Run the training script with the desired number of nodes. For example, to run on 2 nodes,

```sh
DGXNNODES=2 ./flux_training_scale.sh
```