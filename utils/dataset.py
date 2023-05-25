from os.path import splitext
from os import listdir
import os
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as tf
import random
import numpy
import cv2
import albumentations as A

import os
import SimpleITK as sitk
import numpy

def readCT(path):
    image = sitk.ReadImage(path) #读取
    direction=image.GetDirection()
    spacing=image.GetSpacing()
    origin=image.GetOrigin()
    img_arr = sitk.GetArrayFromImage(image)
    return img_arr
    
def writeCT(path, CT, direction=None, spacing=None, origin=None):
    new_image=sitk.GetImageFromArray(CT)
    '''
    new_image.SetSpacing(spacing)
    new_image.SetOrigin(origin)
    new_image.SetDirection(direction)
    '''
    sitk.WriteImage(new_image, path)





w = 800
h = 800

transformer = A.Compose([    
    # 非破坏性转换
    A.VerticalFlip(p=0.2),              
    A.HorizontalFlip(p=0.2),
    A.RandomRotate90(p=0.2),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=118, border_mode=cv2.BORDER_CONSTANT,value=0,mask_value=0, p=0.2),
    # A.Rotate (limit=90, interpolation=1, border_mode=4, value=None, mask_value=None, always_apply=False, p=0.5),
    A.RandomResizedCrop(h, w, scale=(0.35, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=1, always_apply=False, p=0.2),
    
    # 非刚体转换
    # A.OneOf([
    #     A.ElasticTransform(p=0.05, alpha=80, sigma=120 * 0.05, alpha_affine=120 * 0.03, border_mode=cv2.BORDER_CONSTANT,value=0,mask_value=0),
    #     A.GridDistortion(p=0.9, num_steps=15, distort_limit=0.3, border_mode=cv2.BORDER_CONSTANT,value=0,mask_value=0),
    #     A.OpticalDistortion(p=0.05, distort_limit=0.2, shift_limit=0.05, border_mode=cv2.BORDER_CONSTANT,value=0,mask_value=0)                  
    #     ], p=0.9),
    # 非空间性转换
    A.OneOf([
        A.CLAHE(p=0.6),
        #A.RandomBrightnessContrast(brightness_limit = 0.2,contrast_limit = 0.2,p=0.1),    
        A.RandomGamma(p=0.1)], p=0.8),
    
    A.CoarseDropout(max_holes=38, max_height=8, max_width=8,p=0.3),

    A.OneOf([
        A.GaussNoise(p=0.3),
        A.MultiplicativeNoise(p=0.3)], p=0.1),
    
    # A.ImageCompression(quality_lower=70, quality_upper=100, p=0.6),
    A.OneOf([
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.5),
        A.Blur(blur_limit=7, p=0.3),
        A.MedianBlur(blur_limit=3, p=0.1),
        A.MotionBlur(p=0.1),
        A.GaussianBlur(blur_limit=7, p=0.2),
        A.GlassBlur(sigma=0.7, max_delta=1, p=0.1)], p=0.1),
        
    A.OneOf([
        A.Emboss(p=0.3),
        A.Sharpen(p=0.3)], p=0.1),

    A.Resize(512, 512)])






class data_prefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.preload()

    def preload(self):
        try:
            self.next_input0, self.next_input1, self.next_target = next(self.loader)
        except StopIteration:
            self.next_input0 = None
            self.next_input1 = None
            self.next_target = None
            return
        with torch.cuda.stream(self.stream):
            self.next_input0 = self.next_input0.cuda(non_blocking=True).half()
            self.next_input1 = self.next_input1.cuda(non_blocking=True).half()
            self.next_target = self.next_target.cuda(non_blocking=True).long()
            # With Amp, it isn't necessary to manually convert data to half.
            # if args.fp16:
            #     self.next_input = self.next_input.half()
            # else:
            # self.next_input = self.next_input.float()
            # self.next_input = self.next_input.sub_(self.mean).div_(self.std)
            
    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input0 = self.next_input0
        input1 = self.next_input1
        target = self.next_target
        self.preload()
        return input0, input1, target



class BasicDataset(Dataset):
    def __init__(self, 
                 imgs_dir = "/hy-tmp/74e556d0ef5f48c2af338d376fdc054c/CT/Pytorch-UNet-JianBanXian_NiuQu/trainset/data_x", 
                 embed_dir = "/hy-tmp/74e556d0ef5f48c2af338d376fdc054c/CT/Pytorch-UNet-JianBanXian_NiuQu/trainset/embeddings", 
                 masks_dir = "/hy-tmp/74e556d0ef5f48c2af338d376fdc054c/CT/Pytorch-UNet-JianBanXian_NiuQu/trainset/data_y"):
        self.imgs_dir = imgs_dir
        self.embed_dir = embed_dir
        self.masks_dir = masks_dir

        self.mask_file = glob(os.path.join(self.masks_dir, "*.nii.gz"))
        self.img_file = [elm.replace("/data_y/", "/data_x/") for elm in self.mask_file]
        self.embed_file = [elm.replace("/data_y/", "/embeddings/").replace(".nii.gz", "_0.npy") for elm in self.mask_file]
        
        self.numSlice = 64
        




    def __len__(self):
        return len(self.mask_file) * 10
    
    def transform(self, image, mask):
        image5 = transformer(image=image, mask=mask)
        return image5["image"], image5["mask"]
    
    def getPromptData(self, y):
        contours, hierarchy = cv2.findContours(y.astype("uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [elm for elm in contours if cv2.contourArea(elm) > 50]
        if len(contours)> 0:
            # 随机获取y的部分封闭区域
            N_cnts = numpy.random.randint(1, max(2, len(contours)*0.4))
            random.shuffle(contours)
            contours = contours[0:N_cnts]
            Ys = numpy.zeros((N_cnts, y.shape[0], y.shape[1])).astype("uint8")
            prompts = numpy.zeros((N_cnts, y.shape[0], y.shape[1])).astype("uint8")
            # 对每个封闭区域随机选择随机个点，作为指示点
            # print(N_cnts)
            for i in range(N_cnts):
                Ys[i] = cv2.drawContours(Ys[i], contours, i, (1, 1, 1), -1)
                N_cnts = numpy.random.randint(1, 5)
                pointCandidate = numpy.where(Ys[i] > 0)
                # print(len(pointCandidate[0]))
                idxRandom = numpy.random.randint(0, len(pointCandidate[0]), N_cnts)
                pointCandidate = (pointCandidate[0][idxRandom], pointCandidate[1][idxRandom])
                # print(pointCandidate)
                prompts[i][pointCandidate] = 1
            Y = numpy.sum(Ys, 0)
            prompt = numpy.sum(prompts, 0)
            Y = numpy.where(Y > 0, 255, 0)
            prompt = numpy.where(prompt > 0, 255, 0).astype("uint8")        
            prompt = cv2.dilate(prompt, self.kernel, 10)
        else:
            Y, prompt = numpy.zeros((y.shape[0], y.shape[1])).astype("uint8"), numpy.zeros((y.shape[0], y.shape[1])).astype("uint8")
        return Y, prompt

    def __getitem__(self, i):
        i = i % len(self.mask_file)
        #print(self.mask_file[i])
        mask = readCT(self.mask_file[i])
        img = readCT(self.img_file[i])
        embed = numpy.load(self.embed_file[i])
        
        mask = numpy.where(mask > 0, 1, 0)
        
        idx = numpy.random.randint(0, len(mask) - self.numSlice -1)
        
        mask = mask[idx:(idx+self.numSlice)]
        img = img[idx:(idx+self.numSlice)]
        embed = embed[idx:(idx+self.numSlice)]
        
        mask = mask.astype("float32")
        img = img.astype("float32")
        embed = embed.astype("float32")
        # print(mask.min(), mask.max())
        # print(img.min(), img.max())
        # print(embed.min(), embed.max())
        
        img = img[None]
        embed = numpy.transpose(embed, (1,0,2,3))
        
        mask =   torch.from_numpy(mask)
        img =   torch.div(torch.from_numpy(img), 255.0, rounding_mode=None)
        embed =   torch.from_numpy(embed)
        
        return img, embed, mask
        



if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import cv2
    dataset = BasicDataset()
    print(len(dataset))
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    batch = dataset.__getitem__(10)
    
    print(batch[0].size())
    print(batch[1].size())
    print(batch[1].min(), batch[1].max())
    print(batch[2].size())
    print(batch[2].min(), batch[2].max())
    # cv2.imwrite("image.png", (imgs[0].numpy()*255).astype("uint8"))
    # cv2.imwrite("prompt.png", (imgs[1].numpy()*255).astype("uint8"))
    # cv2.imwrite("Y.png", (true_masks[0].numpy()*255).astype('uint8'))

    # for batch in train_loader:
    #     print(batch["image"].shape, batch["mask"].shape)
    #     # cv2.imwrite("image.png", batch["image"][0,0].numpy()*255)
    #     # cv2.imwrite("mask.png", batch["mask"][0,0].numpy()*255)
        
        