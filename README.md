# DLS_project
The reposetory contains results of the final project work for MPTI Deep Learting School 2024 (1 semester)

The aim of the project is to try implement motion blur to a static image. So it is a reverse to image motion debluring task.


# Installation:

Download an `.zip` archive from Google Drive (~2gb):
https://drive.google.com/file/d/1j0lzD40VthaRUp9bANlf8ayFwbv1ELj0/view?usp=sharing

Than run:
```bash
unzip dls_add_motion_blur_project.zip
cd dls_add_motion_blur_project
docker build -t streamlit-app .
```
# Usage:
```bash
docker run -p 8501:8501 streamlit-app
```
# Pipeline
1) Pretrained [DeepLab v3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/):
   
   Input: resized 3 channel input image
   
   Output: segmentation maps for 21 classes
   
3) Segmentation model:
   
   Training data: [ReloBlur dataset](https://leiali.github.io/ReLoBlur_homepage/index.html)
   
   X: resized greyscale transformation of sharp image (from ReloBlur) concatinated with background class map from DeepLab v3
   
   y: blur segmentation map (from ReloBlur)
5) pix2pix GAN:

   Generator:

   X: 4 channel image: 3 RGB channels of resized sharp image (from ReloBlur) + 1 channel blur segmentation map (from segmentation model)
   
   y: 3 channel RGB blur image (from ReloBlur)
   
   Discriminator:
   
   X: 6 channel array: 3 channel RGB blur image (from Generator) + 3 channel RGB blur image (from ReloBlur) - ground truth
   





