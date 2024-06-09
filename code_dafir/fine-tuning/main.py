# from __future__ import print_function
from models_mae import MaskedAutoencoderViT
from models_mae import mae_vit_mini_patch16_dec512d8b
from loss import sequence_loss
from dataset import Dataset
import argparse
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import logging
import numpy as np
import os
from skimage import io
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F
import warnings
warnings.filterwarnings('ignore')

root_path = './'
model_path = './pretrain/epoch_1.pth'

def reduce_tensor(inp):
    """
    Reduce the loss from all processes so that process with rank 0 has the averaged results.
    """
    with torch.no_grad():                    
        reduced_inp = inp                   
        dist.reduce(reduced_inp, dst=0)
    return reduced_inp


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps + 100, pct_start=0.05,
                                              cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler


def reload_pre_train_model(model, device, path=""):
    if not bool(path):
        return model
    else:
        model_dict = model.state_dict()
        pretrained_dict = torch.load(path, map_location='cuda:' + str(device))                                                 


        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

        return model
        
        
def train(train_dataloader, net, optimizer, epoch, device, logger, scheduler, opt):
    net.train()
    running_all, running_loss = 0., 0.
    for batch_idx, (im, bm_gt) in enumerate(train_dataloader, 0):
        
        optimizer.zero_grad()
        im = im.to(device)
        im = net(im)
        bm_gt = bm_gt.to(device)      
       
        loss = (im - bm_gt).abs().mean()

        # for log
        reduced_loss = reduce_tensor(loss)        

        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.module.parameters(), opt.clip)
        optimizer.step()

        scheduler.step()

        batch_size = im.size(0)
        running_all += batch_size
        
        running_loss += reduced_loss.item() * batch_size
        
    # save model
    if opt.local_rank == 0:
        torch.save(net.state_dict(), root_path + 'save/net/epoch_%d.pth' % (epoch))

    # log
    if opt.local_rank == 0:
        logger.info('train: Epoch:{:2}\tbm_loss: {:.8f}'.format(
            epoch,
            running_loss / (running_all * opt.world_size)))   

    return net, scheduler
            


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument('--world_size', default=0)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--batchSize', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--net_if_need_pretrained', default=False)
    parser.add_argument('--net_trained_path', default=root_path + 'save/net/epoch_16.pth')
    parser.add_argument('--manualSeed', type=int, default=1234)

    parser.add_argument('--num_epochs', type=int, default=65)
    parser.add_argument('--num_steps', type=int)
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)

    opt = parser.parse_args()

    dist.init_process_group(backend="nccl", init_method='env://')  

    # logger
    filename = root_path + 'save/log.txt'
    logger_name = "mylog"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(filename, mode='a')
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logger.addHandler(console)

    # random seed
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    if opt.local_rank == 0:
        logger.info("Random Seed: {}".format(opt.manualSeed))
    random.seed(opt.manualSeed)
    np.random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    # world_size or gpu nums
    opt.world_size = torch.distributed.get_world_size()
    if opt.local_rank == 0:
        logger.info("World Size: {}".format(opt.world_size))           ######

    cudnn.benchmark = True

    # train: dataset, dataloader
    train_dataset = Dataset(mode='train')
    train_sampler = DistributedSampler(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=opt.batchSize,  # 32
                                                   shuffle=False,
                                                   num_workers=int(opt.workers),
                                                   pin_memory=True,
                                                   drop_last=True,
                                                   sampler=train_sampler)
                                                   

    # net
    device = torch.device('cuda:{}'.format(opt.local_rank))  
    torch.cuda.set_device(opt.local_rank)
    model = mae_vit_mini_patch16_dec512d8b()
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[opt.local_rank],
                                                output_device=opt.local_rank, find_unused_parameters=True)
    
    # load pretrain  load part weight
    reload_pre_train_model(model, opt.local_rank, model_path)
            
    # setup optimizer
    opt.num_steps = len(train_dataloader) * opt.num_epochs
    if opt.local_rank == 0:
        logger.info('train: images numbers: {:6}\t'.format(len(train_dataset)))              
        logger.info('train: epochs: {:2}\ntrain iters per epoch: {:2}\ntrain total iters: {:2}\n'.format(opt.num_epochs,
                    len(train_dataloader), opt.num_steps))
    optimizer, scheduler = fetch_optimizer(opt, model)

    # train and test
    for epoch in range(opt.num_epochs):
        lr = scheduler.optimizer.param_groups[0]['lr']
        if opt.local_rank == 0:
            logger.info('train: Epoch:{:2}\tLr: {:.8f}\t'.format(epoch + 1, lr))
        train_sampler.set_epoch(epoch)
        model, scheduler = train(train_dataloader, model, optimizer, epoch + 1, device, logger, scheduler, opt)



if __name__ == '__main__':
    main()



