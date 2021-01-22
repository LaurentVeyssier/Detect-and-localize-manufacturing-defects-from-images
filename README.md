# Detect-and-localize-manufacturing-defects-from-images
Use ResNet50 deep learning model to predict defects in steel and visually localize the defect using Res-UNET model.

![](asset/Default.jpg)

This project aims to predict manufacturing defects on steel parts from images. This computer vision technique leverages transfer learning using pretrained ResNet50 model.
In case a default is detected, another model allows to visually show the detected defaults on the image (image segmentation). The second model generates a pixel-wise prediction to localize the defect on the image using a Res-U-net architecture.

# Project description

The project comprizes several steps:
- Classification model to determine whether a steel part has a default of not. The model processes pictures of steel parts and leverages a pre-trained ResNet50 model fined tuned on the target problem using an ad-hoc training dataset. The parts have 4 types of defect however this first step classify parts as faulty / non faulty. Classification of the type of defect is performed during image segmentation step later on.



![](asset/resUnet.jpg)

![](asset/defects.jpg)


![](asset/Unet_architecture.jpg)

![](asset/RLE.jpg)

![](asset/defects.jpg)
