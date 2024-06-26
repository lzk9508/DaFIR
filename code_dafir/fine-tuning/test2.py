# -*- coding: UTF-8 -*-


from models_mae import mae_vit_mini_patch16_dec512d8b
from seg import U2NET

import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as io
import numpy as np
import cv2
import os
import time
from PIL import Image
import argparse
import warnings
warnings.filterwarnings('ignore')


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
        self.net = mae_vit_mini_patch16_dec512d8b()  
        
    def forward(self, x):
        
        bm = self.net(x)
        bm = bm
        bm = 2 * (bm / 255.) - 1  

        return bm      


   

def reload_rec_model(model, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:0')
        print(len(pretrained_dict.keys()))
        pretrained_dict = {k[7:]: v for k, v in pretrained_dict.items() if k[7:] in model_dict}
        print(len(pretrained_dict.keys()))
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
        

def rec(rec_model_path, distorrted_path, save_path, opt):
    print(torch.__version__)
    
    # distorted images list
    img_list = os.listdir(distorrted_path)

    # creat save path for rectified images
    if not os.path.exists(save_path):  
        os.makedirs(save_path)
    
    net = Net(opt).cuda()
    
  
     
    # reload rec model
    reload_rec_model(net.net, rec_model_path)
       
    net = net.eval()
    
    i=0
    print(time.localtime(time.time()))

    for img_path in img_list:
        name = img_path.split('.')[-2]  # image name
        img_path = distorrted_path + img_path  # image path 
        im_ori = np.array(Image.open(img_path))[:, :, :3] / 255.  # read image 0-255 to 0-1
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (512, 512))
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)
        im_s = im_ori.transpose(2, 0, 1)
        im_s = torch.from_numpy(im_s).float().unsqueeze(0)
        
        with torch.no_grad():
            bm = net(im.cuda())         
            bm = bm.detach().cpu()
            print(bm.shape)
            bm0 = cv2.resize(bm[0, 0].numpy(), (w, h))  # x flow
            bm1 = cv2.resize(bm[0, 1].numpy(), (w, h))  # y flow
            bm0 = cv2.blur(bm0, (3, 3))
            bm1 = cv2.blur(bm1, (3, 3))
            lbl = torch.from_numpy(np.stack([bm1, bm0], axis=2)).unsqueeze(0)  # h * w * 2
            


            out = F.grid_sample(im_s, lbl)
            out_img = ((out[0].permute(1, 2, 0).numpy())*255).astype(np.uint8)
            io.imsave(save_path + name + '_rec' + '.png', out_img)
        
        i=i+1
    
    print(time.localtime(time.time()))


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rec_model_path', default='./save/net/')
    parser.add_argument('--distorrted_path', default='/data5/liaozk2/code_dafir/dataset16/')
    parser.add_argument('--index', type=int, default=2)
    opt = parser.parse_args()

    pth_list = sorted(os.listdir(opt.rec_model_path))[-1:]
    print(pth_list)

    for i in pth_list:
        name = i.split('.')[-2] 
        rec(rec_model_path=opt.rec_model_path+i, distorrted_path=opt.distorrted_path, save_path='./test_result3/' + name + '/', opt=opt)


if __name__ == "__main__":
    main()
