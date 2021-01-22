# Detect-and-localize-manufacturing-defects-from-images
Use ResNet50 deep learning model to predict defects in steel and visually localize the defect using Res-UNET model.

![](asset/Default.jpg)

This project aims to predict manufacturing defects on steel parts from images. This computer vision technique leverages transfer learning using pretrained ResNet50 model.
In case a default is detected, another model allows to visually show the detected defaults on the image (image segmentation). The second model generates a pixel-wise prediction to localize the defect on the image using a Res-U-net architecture.

# Project description

The project comprizes two steps:
- Classification model to determine whether a steel part has a default of not. The model processes pictures of steel parts and leverages a pre-trained ResNet50 model fined tuned on the target problem using an ad-hoc training dataset. The parts have 4 types of defect however this first step classifies parts as faulty / non faulty. Determination of the defect type is performed during image segmentation coming next.
- Image segmentation to localize the defects on the image for the faulty parts. This step uses a U-net model to performa a pixel-wise classification, ie predicting whether each pixel of the image is part of the default or not. The output is the visualization of the defect identified by the model. This model is trained on the ad-hoc dataset. The model also generates a prediction for the type of defect between the 4 categories.

Illustration of image segmentation step for faulty parts:

![](asset/resUnet.jpg)

# Dataset

The project uses a training set of 13,000 RGB images (image size of 256 x 1600). Among these, there are over 7,000 images with one of more defects. Defects are classified amongst 4 types with significant class imbalance (predominance of one defect type representing 73% of all defects). This obviously impacts the performance of the model to predict the correct type fo defect. A model predicting all faulty parts to be of type n°3 will mechanically reach a accuracy of 73%. Class imbalance will also influence the learning phase and should be mitigated during training (data augmentation, class weight adjustment).

Dataset struture:

![](asset/defect_prop.jpg)

Steel parts with defect type and localization:

![](asset/defects.jpg)

The training set includes a mask localizing the default on the image for the faulty parts. This mask is encoded using RLE (Run Length Encoding) which is a lossless compression technique to reduce storage requierements. RLE stores sequences containing many consecutive data elements as a single value followed by the count. This is particularly useful to compress image segmentation (binary representation at pixel level with '0' or '1' in our mask example here).

![](asset/RLE.jpg)










![](asset/Unet_architecture.jpg)



![](asset/default.jpg)
