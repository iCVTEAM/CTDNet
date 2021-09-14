# CTDNet
The PyTorch code for ACM MM2021 paper "Complementary Trilateral Decoder for Fast and Accurate Salient Object Detection"

# Requirements
- Python 3.6
- Pytorch 1.4+
- OpenCV 4.0
- Numpy
- TensorboardX
- Apex

# Dataset
Download the SOD datasets and unzip them into ```data``` folder.

# Train
```
cd src
python train.py
```
- We implement our method by PyTorch and conduct experiments on a NVIDIA 1080Ti GPU. 
- We adopt [ResNet-18](https://download.pytorch.org/models/resnet18-5c106cde.pth) and [ResNet-50](https://download.pytorch.org/models/resnet50-19c8e357.pth) pre-trained on ImageNet as backbone networks, which are saved in ```res``` folder.
- We train our method on DUTS-TR and test our method on other datasets.
- After training, the trained models will be saved in ```out``` folder.

# Test
```
cd src
python test.py
```
- After testing, saliency maps will be saved in ```eval``` folder.

# Results
- CTDNet-18： [saliency maps](https://pan.baidu.com/s/1SSZgKFr5EOCaw_hwh_I61A) (提取码：b6ba); [trained model](https://pan.baidu.com/s/1jvprfzYCxPldfFVxWjmWYw) (提取码：ftmz）
- CTDNet-50： [saliency maps](https://pan.baidu.com/s/1mbuyJ7NcSy6mennXMi_iAA) (提取码：j1zq); [trained model](https://pan.baidu.com/s/1HkXvNt0yD-jWtgAeqHYaeg) (提取码：ehvv）

# Evaluation
```
    cd eval
    matlab main
```
- We use  MATLAB code to evaluate the performace of our method.

# Reference
This project is based on the implementation of [F3Net](https://github.com/weijun88/F3Net).
