#from __future__ import print_function
import random
import torch
import torch.utils.data as data
import cv2
import glob
import numpy as np
from PIL import Image


def color_jitter(im, brightness=0., contrast=0., saturation=0., hue=0.):
    f = random.uniform(1 - contrast, 1 + contrast)
    im = np.clip(im * f, 0., 1.)
    f = random.uniform(-brightness, brightness)
    im = np.clip(im + f, 0., 1.).astype(np.float32)

    hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    f = random.uniform(-hue, hue)
    hsv[0] = np.clip(hsv[0] + f * 360, 0., 360.)
    f = random.uniform(-saturation, saturation)
    hsv[2] = np.clip(hsv[2] + f, 0., 1.)
    im = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    im = np.clip(im, 0., 1.)
    return im
    

def data_aug(im):
    im = im / 255.0
    # change background img
    im = color_jitter(im, 0.2, 0.2, 0.2, 0.2)
    return im


class Dataset(data.Dataset): 
    def __init__(self, mode):
        self.mode = mode
        self.img_list = glob.glob('/data/liaozk2/code_dafir/dataset23/data/*.jpg')
        self.img_list.sort()
        self.imagenet_mean = np.array([0.485, 0.456, 0.406])
        self.imagenet_std = np.array([0.229, 0.224, 0.225])
        
        self.bm_list = glob.glob('/data/liaozk2/code_dafir/dataset23/flow/*.npy')
        self.bm_list.sort()
        assert len(self.img_list) == len(self.bm_list)

    def __getitem__(self, index):
        im = np.array(Image.open(self.img_list[index]))[:, :, :3] / 255.  # 0-256
        # deal inputs
#        im = data_aug(im)
        im = (im - self.imagenet_mean) / self.imagenet_std
        im = torch.from_numpy(im).permute(2, 0, 1).float()
        # deal gt
        bm = np.load(self.bm_list[index])  # flow ground truth 256*256*2
        lbl = torch.from_numpy(bm).permute(2, 0, 1).float()    #2*256*256
        return im,lbl

    def __len__(self):
        return len(self.img_list)
