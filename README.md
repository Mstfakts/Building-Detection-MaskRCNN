# Building-Detection-MaskRCNN
The aim of this project is to detect the buildings from the bird's-eye view pictures. I used
[SpaceNet dataset](https://spacenetchallenge.github.io) as a dataset. Because I had limited time,
[matterport's Mask RCNN](https://github.com/matterport/Mask_RCNN) implementation was used not to waste time by coding all the details
of Mask-RCNN.  
Lets get started..

### Project is divided into 5 steps:
#### -Download the dataset.
#### -Understand the Mask-RCNN
#### -İmplement the Mask-RCNN
#### -Preprocess the data
#### -Training & Testing


# 1- Download the SpaceNet Dataset
To download the dataset on your computer, visit the [SpaceNet's website](https://spacenetchallenge.github.io), and check the 'Dependencies'. As it says, you need to have an AWS account, then download the [AWS Command Line Interface (CLI)](https://aws.amazon.com/cli/). In this project, I downloaded the [SpaceNet Buildings Dataset V2](https://spacenetchallenge.github.io/datasets/spacenetBuildings-V2summary.html), but only used 'AOI 2 - Vegas' part of it. Then, copy-paste the commend lines that are shown on the web page to CLI.

# 2- What is Mask-RCNN?
To comprehend what MRCNN is you can read [this paper](https://arxiv.org/abs/1703.06870). 
Here are some helpful websites that helped me out a lot;
- [Image segmentation with Mask R-CNN](https://medium.com/@jonathan_hui/image-segmentation-with-mask-r-cnn-ebe6d793272)
- [Simple Understanding of Mask RCNN](https://medium.com/@alittlepain833/simple-understanding-of-mask-rcnn-134b5b330e95)
- [Computer Vision Tutorial: Implementing Mask R-CNN for Image Segmentation (with Python Code)](https://www.analyticsvidhya.com/blog/2019/07/computer-vision-implementing-mask-r-cnn-image-segmentation/)
