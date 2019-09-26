# Building-Detection-MaskRCNN
The aim of this project is to detect the buildings from the bird's-eye view pictures. I used
[SpaceNet dataset](https://spacenetchallenge.github.io) as a dataset. Because I had limited time,
[matterport's Mask RCNN](https://github.com/matterport/Mask_RCNN) implementation was used not to waste time by coding all the details
of Mask-RCNN.  

Lets get started..

### Project is divided into 5 steps:
#### -Download the SpaceNet Dataset
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

# 3- From Theory to Implementation
Okay, we got the idea of the MRCNN, but how are we going to impelement those theoretical stuff to real code? If you are not concerned with time, please try to implement by yourself. However, I had limited time for this project. So, I used [matterport's Mask RCNN](https://github.com/matterport/Mask_RCNN) implementation. You can search and find other implementations. Do not forget to check the projects that are on matterport's Mask RCNN page. Also, it is possible to search similar projects and read thier codes. It will improve your coding skill.

# 4- Preprocess the Data
After you download the dataset, you can check how geojson files look like by running [Display_GeoJSON.ipynb](https://github.com/Mstfakts/Building-Detection-MaskRCNN/blob/master/Display_GeoJSON.ipynb)


While I was striving to complete this project, I got lots of errors because of the name of the training files. After you download the training dataset, you will notice that the name of files are not in order. I mean, the data includes 'RGB-PanSharpen_AOI_2_Vegas_img1.tif' and 'RGB-PanSharpen_AOI_2_Vegas_img3.tif' but 'RGB-PanSharpen_AOI_2_Vegas_img2.tif'. So, We need to put them in order. To do this, you can run [Rename_Files.ipynb](https://github.com/Mstfakts/Building-Detection-MaskRCNN/blob/master/Rename_Files.ipynb)

We have the dataset and its files' name are in order, but its format it TIFF. So, I converted TIFF file to the RGB file by running
[TIF_to_RGB.ipynb](https://github.com/Mstfakts/Building-Detection-MaskRCNN/blob/master/TIF_to_RGB.ipynb)

To train the model, we need the labels, too. We will create these labels by using TIFF files and its corresponding GeoJSON files. We need the TIFF file to adjust the position of the GeoJSON coordinates to the specific picture. You only need to follow [Create_Masks.ipynb](https://github.com/Mstfakts/Building-Detection-MaskRCNN/blob/master/Create_Masks.ipynb)

If you want to see RGB and its corresponding Mask, run [Display_Mask_and_RGB_Image.ipynb](https://github.com/Mstfakts/Building-Detection-MaskRCNN/blob/master/Display_Mask_and_RGB_Image.ipynb)

# 5- Training the Model & Testing
To train the model, you need to know what size of model your computer can work with. For example, I was using 'resnet101' as a backbone,
but I got OOM (Out Of Memory) error, then I reduced it to the 'resnet50'. If it is possible, try to work with 'resnet101'. Also, do not forget to adjust configuration part regarding to your computer and dataset. Please analyze the [Train.ipynb](https://github.com/Mstfakts/Building-Detection-MaskRCNN/blob/master/Train.ipynb) part. 



That's it. If you need more clarification or have any question please send me an email;

mstfakts98@gmail.com
