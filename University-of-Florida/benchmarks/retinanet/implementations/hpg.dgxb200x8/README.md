Please refer to the implementation directory [b200_pytorch](../b200_pytorch) for more details about data and model preparation.

Copy these files to implementation directory. Modify `retinanet.sh` to set the correct paths for NCCL (from implementation directory), dataset, model, logs, and container.

Run the training script with the desired number of nodes. For example, to run on 2 nodes,

```sh
DGXNNODES=2 ./retinanet.sh
```