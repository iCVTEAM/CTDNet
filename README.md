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

# Model
- If you want to test our method, please download the model into ```out``` folder.
- If you want to train our method, please download the pretrained model into ```res``` folder.

# Train
```shell script
python train.py
```
- We implement our method by PyTorch and conduct experiments on a NVIDIA 1080Ti GPU. 
- We adopt ResNet-18 and ResNet-50 pre-trained on ImageNet as backbone networks, respectively.
- We train our method on DUTS-TR and test our method on other datasets.
- After training, the result models will be saved in ```out``` folder.

# Test
```shell script
python test.py
```
- After testing, saliency maps will be saved in ```eval``` folder.

# Results

# Evaluation

# Reference
This project is based on the implementation of [F3Net](https://github.com/weijun88/F3Net).
