from models_mae import mae_vit_mini_patch16_dec512d8b
#from seg import U2NETP

import torch
import torch.nn as nn
import torch.nn.functional as F
import skimage.io as io
import numpy as np
import cv2
import os
from PIL import Image
import argparse
import warnings
warnings.filterwarnings('ignore')


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


class Net(nn.Module):
    def __init__(self, opt):
        super(Net, self).__init__()
#        self.msk = U2NETP(3, 1)
        self.docmae = mae_vit_mini_patch16_dec512d8b()  # 矫正

    def forward(self, x):
#        msk, _1,_2,_3,_4,_5,_6 = self.msk(x)
#        msk = (msk > 0.5).float()
#        x = msk * x

        x = x.permute(0,2,3,1)
        x[0] = (x[0] - torch.tensor(imagenet_mean).cuda()) / (torch.tensor(imagenet_std).cuda())
        x = x.permute(0,3,1,2)
        loss, y, mask = self.docmae(x)

        return x, loss, y, mask


# def reload_seg_model(model, path=""):
    # if not bool(path):
        # return model
    # else:
        # model_dict = model.state_dict()
        # pretrained_dict = torch.load(path, map_location='cuda:0')
        # print(len(pretrained_dict.keys()))
        # pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if k[6:] in model_dict}
        # print(len(pretrained_dict.keys()))
        # model_dict.update(pretrained_dict)
        # model.load_state_dict(model_dict)

        # return model


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


def rec(mae_model_path, distorrted_path, save_path, opt):
    print(torch.__version__)

    # distorted images list
    img_list = sorted(os.listdir(distorrted_path))
    print(len(img_list))
    # creat save path for rectified images
    if not os.path.exists(save_path):  # 没有路径，创建路径
        os.makedirs(save_path)

    # 实例化
    net = Net(opt).cuda()
    print(get_parameter_number(net))
    # assert 1 == 2

    # reload rec model
    reload_rec_model(net.docmae, mae_model_path)
#    reload_seg_model(net.msk, opt.seg_model_path)

    net.eval()

    for img_path in img_list:
        name = img_path.split('.')[-2]  # image name
        img_path = distorrted_path + img_path  # image path

        im_ori = np.array(Image.open(img_path))[:, :, :3] / 255.  # read image 0-255 to 0-1
        h, w, _ = im_ori.shape
        im = cv2.resize(im_ori, (256, 256))
        # im = im - imagenet_mean
        # im = im / imagenet_std
        im = im.transpose(2, 0, 1)
        im = torch.from_numpy(im).float().unsqueeze(0)

        with torch.no_grad():
            x, loss, y, mask = net(im.cuda())
            y = net.docmae.unpatchify(y)
            y = y.permute(0,2,3,1).detach().cpu()
            x = x.permute(0,2,3,1).detach().cpu()

            mask = mask.detach()
            mask = mask.unsqueeze(-1).repeat(1, 1, net.docmae.patch_embed.patch_size[0] ** 2 * 3)  # (N, H*W, p*p*3)
            mask = net.docmae.unpatchify(mask)  # 1 is removing, 0 is keeping
            mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

            # x = torch.einsum('nchw->nhwc', x)
            im_masked = x * (1 - mask)
            im_paste = x * (1 - mask) + y * mask

            io.imsave(save_path + name + '_original' + '.png', (torch.clamp((x[0] * imagenet_std + imagenet_mean) * 255, 0, 255).numpy()).astype(np.uint8))
            io.imsave(save_path + name + '_masked' + '.png', (torch.clamp((im_masked[0] * imagenet_std + imagenet_mean) * 255, 0, 255).numpy()).astype(np.uint8))
            io.imsave(save_path + name + '_reconstruction' + '.png', (torch.clamp((y[0] * imagenet_std + imagenet_mean) * 255, 0, 255).numpy()).astype(np.uint8))
            io.imsave(save_path + name + '_reconstruction_visible.png', (torch.clamp((im_paste[0] * imagenet_std + imagenet_mean) * 255, 0, 255).numpy()).astype(np.uint8))


def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}


def main():
    parser = argparse.ArgumentParser()
#    parser.add_argument('--seg_model_path', default='/data/wwd/epoch95_iter3644.pth')
    parser.add_argument('--mae_model_path', default='./save/net/epoch_300.pth')
    parser.add_argument('--distorrted_path', default='./test_set/')
    parser.add_argument('--index', type=int, default=0)
    opt = parser.parse_args()

    rec(mae_model_path=opt.mae_model_path,
        distorrted_path=opt.distorrted_path,
        save_path='./test_result/',
        opt=opt)

if __name__ == "__main__":
    main()
