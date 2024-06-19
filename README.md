## MULDE: Multiscale Log-Density Estimation via Denoising Score Matching for Video Anomaly Detection

This is the official PyTorch implementation of the density-based anomaly detector "MULDE" which is trained via score matching. The anomaly detector is proposed in the paper 
[MULDE: Multiscale Log-Density Estimation via Denoising Score Matching for Video Anomaly Detection](https://openaccess.thecvf.com/content/CVPR2024/html/Micorek_MULDE_Multiscale_Log-Density_Estimation_via_Denoising_Score_Matching_for_Video_CVPR_2024_paper.html) presented at CVPR 2024.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mulde-multiscale-log-density-estimation-via/anomaly-detection-on-ucsd-ped2)](https://paperswithcode.com/sota/anomaly-detection-on-ucsd-ped2?p=mulde-multiscale-log-density-estimation-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mulde-multiscale-log-density-estimation-via/anomaly-detection-on-chuk-avenue)](https://paperswithcode.com/sota/anomaly-detection-on-chuk-avenue?p=mulde-multiscale-log-density-estimation-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mulde-multiscale-log-density-estimation-via/anomaly-detection-on-shanghaitech)](https://paperswithcode.com/sota/anomaly-detection-on-shanghaitech?p=mulde-multiscale-log-density-estimation-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mulde-multiscale-log-density-estimation-via/anomaly-detection-on-ubnormal)](https://paperswithcode.com/sota/anomaly-detection-on-ubnormal?p=mulde-multiscale-log-density-estimation-via)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/mulde-multiscale-log-density-estimation-via/anomaly-detection-in-surveillance-videos-on)](https://paperswithcode.com/sota/anomaly-detection-in-surveillance-videos-on?p=mulde-multiscale-log-density-estimation-via)

[![MULDE-short-video](https://img.youtube.com/vi/WoYCf9NWQmQ/0.jpg)](https://www.youtube.com/watch?v=WoYCf9NWQmQ)

## Install

We recommend to run the code in a virtual environment or a conda environment.

To install the required packages, run the following commands:
```
conda env create -f environment.yml
conda activate mulde
```


## Usage
To run the training and evaluation pipeline with visualizations of the densities and the dataset, run the following command:
```
python main.py --plot_dataset --gmm
```

If playing around with training/testing skip the ``--gmm`` option as it is implemented in sklearn and can take some time to fit the GMM to the data. 

To set the parameters explained in the paper, you can set them appropriately. A selection of the essential parameters are as follows:
```
python main.py --plot_dataset --gmm --beta 0.1 --L 16 --sigma_low 1e-3 --sigma_high 1. --units 4096 4096 --lr 4e-5 --batch_size 2048 --epochs 1000
```


## Visualizations
To view visualizations run the tensorboard command:
```
tensorboard --logdir=runs/MULDE --samples_per_plugin images=100
```
In the tensorboard tab "Scalars" you can view the anomaly scores on each individual scale under ``roc_auc_*_individual``. The aggregates like max/median/mean and the negative log likelihood scores of the GMM are in the ``roc_auc_*_aggregate``. Under ``_roc_auc_best`` you can find the best performing individual scales or aggregates respectively.

Under the "Images" tab you can find and inspect the visualizations and comparisons of the density maps and score norm maps.

## Citation
If you find this work useful, please consider citing:
```
@inproceedings{micorek24mulde,
  title = {{MULDE: Multiscale Log-Density Estimation via Denoising Score Matching for Video Anomaly Detection}},
  author = {Micorek, Jakub and Possegger, Horst and Narnhofer, Dominik and Bischof, Horst and Kozi{\'n}ski, Mateusz},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {18868-18877}
}
```
