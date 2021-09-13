# CTDNet
Code for ACM MM2021 paper "Complementary Trilateral Decoder for Fast and Accurate Salient Object Detection"

The PyTorch implementation of the CTDNet.

# Requirements
- [Python 3.6](https://www.python.org/)
- [Pytorch 1.0+](http://pytorch.org/)
- [OpenCV 4.0](https://opencv.org/)
- [Numpy](https://numpy.org/)
- [TensorboardX](https://github.com/lanpa/tensorboardX)
- [Apex](https://github.com/NVIDIA/apex)

# Dataset
Download the SOD datasets and unzip them into data folder.

# Model
- If you want to test our method, please download the model into out folder.
- If you want to train our method, please download the pretrained model into res folder.

# Train
```shell script
python train.py
```
- We implement our method by PyTorch and conduct experiments on a NVIDIA 1080Ti GPU. 
- We adopt ResNet-18 and ResNet-50 pre-trained on ImageNet as backbone networks, respectively.
- We train our method on DUTS-TR and test our method on other datasets.
- After training, the result models will be saved in out folder.

# Test
```shell script
python test.py
```
- After testing, saliency maps will be saved in eval folder.

# Results

# Evaluation
