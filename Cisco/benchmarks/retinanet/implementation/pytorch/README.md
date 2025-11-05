## Running NVIDIA SSD PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA SSD PyTorch MLPerf Benchmark on NVIDIA hardware.

## 1. Hardware Requirements

- At least 1TB disk space is required.
- NVIDIA GPU with at least 80GB memory is strongly recommended.
- GPU is not needed for preprocessing scripts, but is needed for training.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:single_stage_detector-pyt
```

### 3.2 Download dataset

Start the container, replacing `</path/to/your/data>` with the existing path to where you want to save the SSD dataset:

```bash
docker run -it --gpus=all --runtime=nvidia --ipc=host -v </path/to/your/data>:/workspace/ssd_dataset <docker/registry>/mlperf-nvidia:single_stage_detector-pyt
```

The benchmark uses a subset of [OpenImages-v6](https://storage.googleapis.com/openimages/web/index.html).

To download the subset:

```bash
pip install fiftyone==0.15.1
cd /workspace/ssd/public-scripts
./download_openimages_mlperf.sh -d /workspace/ssd_dataset
```

The script will download the benchmark subset with metadata and labels, then
convert the labels to [COCO](https://cocodataset.org/#home) format. The
downloaded dataset size is 352GB and the expected folder structure after
running the script is:

```
/workspace/ssd_dataset
│
└───info.json
│
└───train
│   └─── data
│   │      000002b66c9c498e.jpg
│   │      000002b97e5471a0.jpg
│   │      ...
│   └─── metadata
│   │      classes.csv
│   │      hierarchy.json
│   │      image_ids.csv
│   └─── labels
│          detections.csv
│          openimages-mlperf.json
│
└───validation
    └─── data
    │      0001eeaf4aed83f9.jpg
    │      0004886b7d043cfd.jpg
    │      ...
    └─── metadata
    │      classes.csv
    │      hierarchy.json
    │      image_ids.csv
    └─── labels
           detections.csv
           openimages-mlperf.json
```

Read more about the mlperf subset [here](#the-mlperf-subset).

### 3.3 Download the pretrained backbone
The benchmark uses a ResNeXt50_32x4d backbone pretrained on ImageNet. The
weights are downloaded from PyTorch hub.

By default, the code will automatically download the weights to
`$TORCH_HOME/hub` (default is `~/.cache/torch/hub`) and save them for later use.

Alternatively, you can manually download the weights with:
```bash
cd /workspace/ssd
bash ./public-scripts/download_backbone.sh
```

Then use the downloaded file with `--pretrained <PATH TO WEIGHTS>` .

## 4. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:
```bash
export DATADIR="<path/to/dir/containing/openimages/dir>"
export BACKBONE_DIR="<path/to/pretrained/ckpt>"
export LOGDIR="<path/to/output/dir>"
export CONT=<docker/registry>/mlperf-nvidia:single_stage_detector-pyt
source config_DGXH100_001x08x032.sh  # select config and source it
sbatch -N $DGXNNODES -t $WALLTIME run.sub  # you may be required to set --account and --partition here
```

Hyperparameters are recorded in the `config_*.sh` files for each configuration and in `run_and_time.sh`.

## 5. Dataset 

### 5.1 Publication/Attribution
[Google Open Images Dataset V6](https://storage.googleapis.com/openimages/web/index.html)

### 5.2 The MLPerf Subset
The MLPerf subset includes only 264 classes of the total 601 available in the
full dataset:

| Dataset           | # classes | # train images | # validation images | Size  |
|-------------------|-----------|----------------|---------------------|-------|
| OpenImages Full   | 601       | 1,743,042      | 41,620              | 534GB |
| OpenImages MLperf | 264       | 1,170,301      | 24,781              | 352GB |

These are the lowest level classes (no child classes) in the dataset
[semantic hierarchy tree](https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy_visualizer/circle.html)
with at least 1000 samples.

The list of used classes can be viewed
[here](https://github.com/mlcommons/training/blob/master/single_stage_detector/scripts/download_openimages_mlperf.sh).

## 6. Model
This network takes an input 800x800 image from
[OpenImages-v6](https://storage.googleapis.com/openimages/web/index.html)
and 264 categories, and computes a set of bounding boxes and categories.
Other detector models use multiple stages, first proposing regions of interest
that might contain objects, then iterating over the regions of interest to try
to categorize each object. SSD does both of these in one stage, leading to
lower-latency and higher-performance inference.

### 6.1 Backbone

The backbone is based on ResNeXt50_32x4d as described in Section 3 of
[this paper](https://arxiv.org/pdf/1611.05431.pdf).  Using the
same notation as Table 1 of the paper the backbone looks like:

| stage      | # stacked blocks | shape of a residual block  |
| :--------: | :--------------: | :------------------------: |
| conv1      |                  | 7x7, 64, stride 2          |
|            |                  | 3x3 max pool, stride 2     |
| conv2_x    | 3                | 1x1, 128                   |
|            |                  | 3x3, 128, groups=32        |
|            |                  | 1x1, 256                   |
| conv3_x    | 4                | 1x1, 256                   |
|            |                  | 3x3, 256, groups=32        |
|            |                  | 1x1, 512                   |
| conv4_x    | 6                | 1x1, 512                   |
|            |                  | 3x3, 512, groups=32        |
|            |                  | 1x1, 1024                  |
| conv5_x    | 3                | 1x1, 1024                  |
|            |                  | 3x3, 1024, groups=32       |
|            |                  | 1x1, 2048                  |

Input images are 800x800 RGB. They are fed to a 7x7 stride 2 convolution with
64 output channels, then through a 3x3 stride 2 max-pool layer.
The rest of the backbone is built from "building blocks": 3x3
grouped convolutions with a "short-cut" residual connection
around the pair.  All convolutions in the backbone are followed by batch-norm
and ReLU.

The backbone is initialized with the pretrained weights from the corresponding
layers of the ResNeXt50_32x4d implementation from the [Torchvision model
zoo](https://download.pytorch.org/models/-7cdf4587.pth), described in
detail [here](https://pytorch.org/hub/pytorch_vision_resnext/).  It is
a ResNeXt50_32x4d network trained on 224x224 ImageNet to achieve a Top-1
error rate of 22.38  and a Top-5 error rate of 6.30.

Of the five convolution stages, only the last three are trained.
The weights of the first two stages are frozen
([code](https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/model/backbone_utils.py#L94-L101)).
In addition, all batch norm layers in the backbone are frozen
([code](https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/model/backbone_utils.py#L52)).


### 6.2 Weight and bias initialization
1. The ResNeXt50_32x4d backbone is initialized with the pretrained weights
   from [Torchvision model zoo](https://download.pytorch.org/models/-7cdf4587.pth).

2. The classification head weights are initialized using normal distribution
   with `mean=0` and `std=0.01`. The biases are initialized with zeros, except
   for the classification convolution which is initialized with
   `constant=-4.59511985013459`
   ([code](https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/model/retinanet.py#L85-L90)).

3. The regression head weights are initialized using normal distribution
   with `mean=0` and `std=0.01`. The biases are initialized with zeros
   ([code](https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/model/retinanet.py#L171-L177)).

4. The FPN network weights are initialized with uniform Kaiming (also known as
   He initialization) using `negative slope=1`. The biases are initialized
   with zeros
   ([code](https://github.com/mlcommons/training/blob/master/single_stage_detector/ssd/model/feature_pyramid_network.py#L90-L91)).


### 6.3 Input augmentations
The input images are assumed to be sRGB with values in range 0.0 through 1.0.
The input pipeline does the following:

1. Random horizontal flip of both the image and its ground-truth bounding boxes
   with a probability of 50%.

2. Normalize the colors to a mean of (0.485, 0.456, 0.406) and standard
   deviation (0.229, 0.224, 0.225).

3. Resize image to 800x800 using bilinear interpolation.


### 6.4 Publication/Attribution

Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, Scott Reed,
Cheng-Yang Fu, Alexander C. Berg.  [SSD: Single Shot MultiBox
Detector](https://arxiv.org/abs/1512.02325). In the _Proceedings of the
European Conference on Computer Vision_, (ECCV-14):21-37, 2016.

Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun.  [Deep Residual Learning for
Image Recognition](https://arxiv.org/abs/1512.03385).  In the _Proceedings of
the Conference on Computer Vision and Pattern Recognition_, (CVPR):770-778, 2016.

Jonathan Huang, Vivek Rathod, Chen Sun, Menglong Zhu, Anoop Korattikara,
Alireza Fathi, Ian Fischer, Zbigniew Wojna, Yang Song, Sergio Guadarrama, Kevin
Murphy. [Speed/accuracy trade-offs for modern convolutional object
detectors](https://arxiv.org/abs/1611.10012).  In the _Proceedings of the
Conference on Computer Vision and Pattern Recognition_, (CVPR):3296-3305, 2017.

Krasin I., Duerig T., Alldrin N., Ferrari V., Abu-El-Haija S., Kuznetsova A.,
Rom H., Uijlings J., Popov S., Kamali S., Malloci M., Pont-Tuset J., Veit A.,
Belongie S., Gomes V., Gupta A., Sun C., Chechik G., Cai D., Feng Z.,
Narayanan D., Murphy K.
[OpenImages](https://storage.googleapis.com/openimages/web/index.html): A public
dataset for large-scale multi-label and multi-class image classification, 2017.

Saining Xie, Ross Girshick, Piotr Dollár, Zhuowen Tu, Kaiming He.
[Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/abs/1611.05431)

Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár.
[Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002)

Torchvision pretrained [ResNeXt50_32x4d](https://pytorch.org/vision/0.12/models.html#id25) on ImageNet

Torchvision [RetinaNet](https://pytorch.org/vision/0.12/models.html#id65)

## 7. Quality

### 7.1 Quality metric
Metric is COCO box mAP (averaged over IoU of 0.5:0.95), computed over the
OpenImages-MLPerf validation subset.

### 7.2 Quality target
mAP of 0.34

### 7.3 Evaluation frequency
Every epoch, starting with the first one.

### 7.4 Evaluation thoroughness
All the images in the OpenImages-MLPerf validation subset
