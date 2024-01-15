# DRConv

This repo is the official implementation of [Dynamic region-aware convolution](https://openaccess.thecvf.com/content/CVPR2021/papers/Chen_Dynamic_Region-Aware_Convolution_CVPR_2021_paper.pdf)

We propose a new convolution called Dynamic Region-Aware Convolution (DRConv), which can automatically assign multiple filters to corresponding spatial regions where features have similar representation. In this way, DRConv outperforms standard convolution in modeling semantic variations. Standard convolutional layer can increase the number of filers to extract more visual elements but results in high computational cost. More gracefully, our DRConv transfers the increasing channel-wise filters to spatial dimension with learnable instructor, which not only improves representation ability of convolution, but also maintains computational cost and the translation-invariance as standard convolution dose. DRConv is an effective and elegant method for handling complex and variable spatial information distribution. It can substitute standard convolution in any existing networks for its plug-and-play property, especially to power convolution layers in efficient networks. We evaluate DRConv on a wide range of models (MobileNet series, ShuffleNetV2, etc.) and tasks (Classification, Face Recognition, Detection and Segmentation). On ImageNet classification, DRConv-based ShuffleNetV2-0.5x achieves state-of-the-art performance of 67.1% at 46M multiply-adds level with 6.3% relative improvement.

## Compile cuda version of DRConv
    - `cd Deformable MatMul3`
    - `bash compile.sh`


## How to use DRConv
Take ShuffleNetV2 for example, refer to ShuffleNetV2_pytorch.py for details.

## License
DRConv is released under the Apache 2.0 license.




