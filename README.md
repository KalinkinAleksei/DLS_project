# DLS_project
The reposetory contains results of the final project work for MPTI Deep Learting School 2024 (1 semester)

The aim of the project is to try implement motion blur to a static image. So it is a reverse to image motion debluring task.


# Installation:

Download a `.zip` archive from Google Drive (~2gb):
https://drive.google.com/file/d/1j0lzD40VthaRUp9bANlf8ayFwbv1ELj0/view?usp=sharing

Than run:
```bash
unzip dls_add_motion_blur_project.zip
cd dls_add_motion_blur_project
docker build -t dls-blur-project .
```
# Usage:
```bash
docker run -p 8501:8501 dls-blur-project
```
# Pipeline:
**1) Pretrained [DeepLab v3](https://pytorch.org/hub/pytorch_vision_deeplabv3_resnet101/):**
   
   - **Input**: resized 3 channel input image
   
   - **Output**: segmentation maps for 21 classes
   
**2) Segmentation model:**
   
   - **Training data:** [ReloBlur dataset](https://leiali.github.io/ReLoBlur_homepage/index.html)
   
   - **X:** 2-cannel array: 1-channel resized grayscale transformation of sharp image (from ReloBlur) + 1-channel background class map from DeepLab v3
   
   - **y:** 1-channel blur segmentation map (from ReloBlur)
     
**3) pix2pix GAN:**

   - **Generator:**

     **X:** 4-channel array: 3-channel resized RGB sharp image (from ReloBlur) + 1-channel blur segmentation map (from segmentation model)

     **y:** 3-channel RGB blur image (from ReloBlur)
   
   - **Discriminator:**

     **X:** 6-channel array: 3-channel RGB blur image (from Generator) + 3-channel RGB blur image (from ReloBlur) - ground truth






