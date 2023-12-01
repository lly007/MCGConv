# MCGConv: Multi-Graph Aggregation via Various Centroids for Deep Learning on Point Cloud

**MCGConv is a spatial convolution operator for point cloud**


![The Overview of MCGConv](https://github.com/lly007/MCGConv/blob/main/pic/overview.png "The Overview of FPAC")


This is an implementation of MCGConv by PyTorch.


# Introduction

This project propose a new scheme, called Multi-Centroids Graph Convolution (MCGConv), for performing the 3D point cloud convolution and extracting the features from the individual cloud points.



##  Environment
This project passed the test in the following environment
### Software
- PyTorch >= 2.1.0
- PyTorch3D >= 0.7.5 [how to install](https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md)
- NVIDIA CUDA Toolkit >= 12.1

### Harware
- NVIDIA TITAN RTX / NVIDIA Tesla V100 / NVIDIA GeForce RTX3090
- 32GB RAM



## Classification
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

### Run
```shell
python train_classification_MCGConv_tb.py --normal --log_dir [your log dir]
```

## Part Segmentation
### Data Preparation
Download alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip)  and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal/`.
### Run
```shell
python train_partseg_MCGConv_tb.py --normal --log_dir [your log dir]
```

## Semantic Segmentation
### Data Preparation
Download 3D indoor parsing dataset (**S3DIS**) [here](http://buildingparser.stanford.edu/dataset.html)  and save in `data/Stanford3dDataset_v1.2_Aligned_Version/`.
```shell
cd data_utils
python collect_indoor3d_data.py
```
Processed data will save in `data/stanford_indoor3d/`.
### Run
```shell
python train_semseg.py --log_dir [your log dir]
python test_semseg.py --log_dir [your log dir] --test_area 5 --visual
```

## Experiment Reference
This implementation of experiment is heavily reference to [yanx27/Pointnet_Pointnet2_pytorch](https://github.com/yanx27/Pointnet_Pointnet2_pytorch)<br>
Thanks very much !
