# from __future__ import print_function
#from model import IDGR
#from dataset import Dataset

import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import logging
import numpy as np
import torch.nn.functional as F

import warnings
warnings.filterwarnings('ignore')

root_path = '/data5/liaozk2/PCN-main/new_task4/'


def sequence_loss(flow_preds, flow_gt, gamma=0.8):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
#    print(len(flow_preds))
#    print(flow_gt[0].shape)
    flow_loss = 0.0
    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
#        msk = ((flow_gt[:, 0] != 0)*(flow_gt[:, 1]!=0)) * 1.  # b*h*w, 因为flow_gt[:, 0]的尺寸是b*h*w, 把第二个通道的0单独拎出来。
#        msk = msk.unsqueeze(1) # b*1*h*w，在1处增加一个维度。
        # flow_preds[i]=torch.mul(flow_preds[i], msk)
        # print(flow_preds.shape)        #shape不用加括号，size才要。
        # print(flow_gt.shape)
        # input()
        i_loss = (flow_preds[i] - flow_gt).abs()
              
        flow_loss += i_weight * i_loss.mean() 

    return flow_loss


