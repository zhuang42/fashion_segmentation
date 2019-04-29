### CSC420 Project 3 Fashion Segmentation

### Introduction

This is for Project 3 for the CSC420 class: http://www.cs.utoronto.ca/~fidler/teaching/2015/CSC420.html

The data is adapted from the following paper:

Kota Yamaguchi, M Hadi Kiapour, Luis E Ortiz, Tamara L Berg, "Parsing Clothing in Fashion Photographs", CVPR 2012
http://vision.is.tohoku.ac.jp/~kyamagu/research/clothing_parsing/

Explanation of the data:

- data/fashion_person/*/*.jpg  train and val images
- data/fashion_person/*/*_person.png   contains image labeling into person and background. If pixel has value 1, it belongs to the person class, otherwise it is background

- data/fashion_clothes/*/*.jpg  train and val images
- data/fashion_clothes/*/*_clothes.png  contain image labeling for 6 clothing types and background. See labels.txt for the label information.

The models we used is [Unet], [DeepLab-V3] and [DeepLab-V3-Plus](https://arxiv.org/pdf/1802.02611) with Resnet Backbone.


### Dependency:
Environment: Python 3.6, Pytorch 1.0.1, CUDA 9.0, GTX 1060 6GB.

conda install numpy
conda install tqdm
conda install pytorch torchvision -c pytorch
pip install torchsummary
conda install -c conda-forge tensorboardx
conda install scipy
conda install matplotlib
conda install -c conda-forge scikit-image

### Usage of predict.py:
usage: predict.py [-h] [--model [deeplabv3+, deeplabv3, unet]] --task
[person, fashion] [--path model_path]
--input input_path --output output_path
optional arguments:
-h, --help show this help message and exit
--model [deeplabv3+, deeplabv3, unet], -m [deeplabv3+, deeplabv3,
unet]
Specify Which Model(default : DeepLabV3+)
--task [person, fashion], -t [person, fashion]
Specify Task [person, fashion]
--path model_path, -p model_path
Specify Model Path
--input input_path, -i input_path
Input image
--output output_path, -o output_path
Output image
Example:python predict.py -t person -i ./image.jpg -o output.png
It may takes 3-4 seconds run on CPU.

### Acknowledgement
[pytorch-deeplab-xception]https://github.com/jfzhang95/pytorch-deeplab-xception)


### Reference:
1. Kota Yamaguchi, M Hadi Kiapour, Luis E Ortiz, Tamara L Berg,
“Parsing Clothing in Fashion Photographs”, CVPR 2012. http://
vision.is.tohoku.ac.jp/~kyamagu/research/clothing_parsing/
2. Kaiming He, Xiangyu Zhang, Shaoqing Ren: “Deep Residual
Learning for Image Recognition”, 2015; arXiv:1512.03385.
3. Olaf Ronneberger, Philipp Fischer: “U-Net: Convolutional Networks
for Biomedical Image Segmentation”, 2015; arXiv:1505.04597.
4. Liang-Chieh Chen, George Papandreou, Iasonas Kokkinos, Kevin
Murphy: “DeepLab: Semantic Image Segmentation with Deep
Convolutional Nets, Atrous Convolution, and Fully Connected CRFs”,
2016; arXiv:1606.00915.
5. Jonathan Long, Evan Shelhamer: “Fully Convolutional Networks for
Semantic Segmentation”, 2014; arXiv:1411.4038.
6. Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff:
“Encoder-Decoder with Atrous Separable Convolution for Semantic
Image Segmentation”, 2018; arXiv:1802.02611.
7. Liang-Chieh Chen, George Papandreou, Florian Schroff: “Rethinking
Atrous Convolution for Semantic Image Segmentation”, 2017; arXiv:
1706.05587.
8. pytorch-deeplab-xception
https://github.com/jfzhang95/pytorch-deeplab-xception
9. “Why Data Normalization is necessary for Machine Learning
models” : https://medium.com/@urvashilluniya/why-datanormalization-
is-necessary-for-machine-learningmodels-
681b65a05029
10.Evaluating image segmentation models.
https://www.jeremyjordan.me/evaluating-image-segmentationmodels/


