import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import torchvision.models as models
import streamlit as st
from skimage.io import imread
from skimage.transform import resize
from skimage.color import rgb2gray
from PIL import Image
import numpy as np

st.title('Add motion blur project')

deeplab = models.segmentation.deeplabv3_resnet50(pretrained=False)
deeplab.load_state_dict(torch.load('deeplabv3_resnet50.pth'), strict=False)

def get_background(image, deeplab):
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.float().unsqueeze(0)
    deeplab.eval()
    with torch.no_grad():
        output = deeplab(input_batch)['out'][0]
    output = output.numpy()
    max_indices = np.argmax(output, axis=0)
    background = np.array([np.where(max_indices == 0, 0, 1)])
    return background

def get_map(im_for_segmentation, seg_model):
    seg_model.eval()
    X = torch.from_numpy(im_for_segmentation).float()
    output = torch.sigmoid(seg_model(X.view(1, 2, 256, 256)))
    output = output.detach().numpy()
    return output[0][0]

class SegNet(nn.Module):
    def __init__(self, in_channels=2, out_channels=1):
        super().__init__()

        self.in_chn = in_channels
        self.out_chn = out_channels

        # VGG-16 architecture
        self.enc_conv01 = nn.Conv2d(self.in_chn, 64, kernel_size=3, padding=1)
        self.enc_bn01 = nn.BatchNorm2d(64)
        self.enc_conv02 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.enc_bn02 = nn.BatchNorm2d(64)
        self.pool0 = nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True)

        self.enc_conv11 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_bn11 = nn.BatchNorm2d(128)
        self.enc_conv12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.enc_bn12 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True)

        self.enc_conv21 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_bn21 = nn.BatchNorm2d(256)
        self.enc_conv22 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.enc_bn22 = nn.BatchNorm2d(256)
        self.enc_conv23 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.enc_bn23 = nn.BatchNorm2d(256)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True)

        self.enc_conv31 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc_bn31 = nn.BatchNorm2d(512)
        self.enc_conv32 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc_bn32 = nn.BatchNorm2d(512)
        self.enc_conv33 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc_bn33 = nn.BatchNorm2d(512)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=None, return_indices=True)

        # bottleneck
        self.bottleneck_conv0 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bottleneck_bn0 = nn.BatchNorm2d(512)
        self.bottleneck_conv1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bottleneck_bn1 = nn.BatchNorm2d(512)

        # decoder
        self.upsample0 = nn.MaxUnpool2d(kernel_size=2)
        self.dec_conv01 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_bn01 = nn.BatchNorm2d(512)
        self.dec_conv02 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_bn02 = nn.BatchNorm2d(512)
        self.dec_conv03 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.dec_bn03 = nn.BatchNorm2d(256)

        self.upsample1 = nn.MaxUnpool2d(kernel_size=2)
        self.dec_conv11 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_bn11 = nn.BatchNorm2d(256)
        self.dec_conv12 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_bn12 = nn.BatchNorm2d(256)
        self.dec_conv13 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.dec_bn13 = nn.BatchNorm2d(128)

        self.upsample2 = nn.MaxUnpool2d(kernel_size=2)
        self.dec_conv21 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec_bn21 = nn.BatchNorm2d(128)
        self.dec_conv22 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.dec_bn22 = nn.BatchNorm2d(64)

        self.upsample3 = nn.MaxUnpool2d(kernel_size=2)
        self.dec_conv31 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec_bn31 = nn.BatchNorm2d(64)
        self.dec_conv32 = nn.Conv2d(64, self.out_chn, kernel_size=3, padding=1)

    def forward(self, x):
        # encoder
        e01 = F.relu(self.enc_bn01(self.enc_conv01(x)))
        e02 = F.relu(self.enc_bn02(self.enc_conv02(e01)))
        e0, id0 = self.pool0(e02)

        e11 = F.relu(self.enc_bn11(self.enc_conv11(e0)))
        e12 = F.relu(self.enc_bn12(self.enc_conv12(e11)))
        e1, id1 = self.pool1(e12)

        e21 = F.relu(self.enc_bn21(self.enc_conv21(e1)))
        e22 = F.relu(self.enc_bn22(self.enc_conv22(e21)))
        e23 = F.relu(self.enc_bn23(self.enc_conv23(e22)))
        e2, id2 = self.pool2(e23)

        e31 = F.relu(self.enc_bn31(self.enc_conv31(e2)))
        e32 = F.relu(self.enc_bn32(self.enc_conv32(e31)))
        e33 = F.relu(self.enc_bn33(self.enc_conv33(e32)))
        e3, id3 = self.pool3(e33)

        # bottleneck
        b0 = F.relu(self.bottleneck_bn0(self.bottleneck_conv0(e3)))
        b1 = F.relu(self.bottleneck_bn1(self.bottleneck_conv1(b0)))

        # decoder
        d00 = self.upsample0(b1, id3)
        d01 = F.relu(self.dec_bn01(self.dec_conv01(d00)))
        d02 = F.relu(self.dec_bn02(self.dec_conv02(d01)))
        d03 = F.relu(self.dec_bn03(self.dec_conv03(d02)))

        d10 = self.upsample1(d03, id2)
        d11 = F.relu(self.dec_bn11(self.dec_conv11(d10)))
        d12 = F.relu(self.dec_bn12(self.dec_conv12(d11)))
        d13 = F.relu(self.dec_bn13(self.dec_conv13(d12)))

        d20 = self.upsample2(d13, id1)
        d21 = F.relu(self.dec_bn21(self.dec_conv21(d20)))
        d22 = F.relu(self.dec_bn22(self.dec_conv22(d21)))

        d30 = self.upsample3(d22, id0)
        d31 = F.relu(self.dec_bn31(self.dec_conv31(d30)))
        d32 = self.dec_conv32(d31)

        return d32

seg_model = torch.load('seg_model.pth')
seg_model.cpu()
seg_model.eval()


class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat([x, skip_input], 1)
        return x
class UNetGenerator(nn.Module):
    def __init__(self, input_channels=4, output_channels=3):
        super().__init__()
        # Increased channel sizes by ~1.5x
        self.down1 = UNetDown(input_channels, 192, normalize=False)
        self.down2 = UNetDown(192, 384)
        self.down3 = UNetDown(384, 768)
        self.down4 = UNetDown(768, 1536, dropout=0.5)
        self.down5 = UNetDown(1536, 1536, dropout=0.5)
        self.down6 = UNetDown(1536, 1536, dropout=0.5)
        self.down7 = UNetDown(1536, 1536, dropout=0.5)
        self.down8 = UNetDown(1536, 1536, normalize=False, dropout=0.5)

        # Adjusted upsampling layers for increased channels
        self.up1 = UNetUp(1536, 1536, dropout=0.5)
        self.up2 = UNetUp(3072, 1536, dropout=0.5)
        self.up3 = UNetUp(3072, 1536, dropout=0.5)
        self.up4 = UNetUp(3072, 1536, dropout=0.5)
        self.up5 = UNetUp(3072, 768)
        self.up6 = UNetUp(1536, 384)
        self.up7 = UNetUp(768, 192)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(384, output_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)
    
generator = torch.load('generator_v1.pth')
generator.cpu()
generator.eval()


file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if file is not None:
    image = imread(file)
    img_display = Image.fromarray(image)
    st.image(img_display, caption='Input Image',use_container_width=True)

    im = resize(image, (256, 256), mode='constant', anti_aliasing=False,)
    bg = get_background(im, deeplab)
    bg_normalized = (bg * 255).astype(np.uint8)
    bg_normalized = bg_normalized.squeeze()
    st.image(bg_normalized, caption='Background by DeepLabv3',use_container_width=True)
    
    #Seg model
    im_grey = np.array([rgb2gray(im)])
    im_for_segmentation = np.concatenate((im_grey, bg), axis=0)
    mp = np.array([get_map(im_for_segmentation, seg_model)])
    mp_normalized = (mp * 255).astype(np.uint8)
    mp_normalized = mp_normalized.squeeze()
    st.image(mp_normalized, caption='Possible blur segmentation',use_container_width=True)

    #GAN
    image_final = np.transpose(im, (2, 0, 1))
    image_final = np.concatenate((image_final, mp), axis=0)
    X = torch.from_numpy(np.array([image_final])).float()
    pred = generator(X)
    result = pred.detach().numpy()
    result = result.squeeze().transpose(1, 2, 0)
    result = (result*255).astype(np.uint8)
    st.image(result, caption='generated image',use_container_width=True)

