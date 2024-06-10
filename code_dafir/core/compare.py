import cv2 as cv
import glob
import time
import os
from metric import ssim, psnr, mae
import logging   
import warnings
warnings.filterwarnings('ignore')

ssim_now=0
psnr_now=0
mae_now=0


root_path = '/code_dafir/fine-tuning/test_result/'

if __name__ == "__main__":
  # logger
  filename = root_path + 'log.txt'
  logger_name = "mylog"
  logger = logging.getLogger(logger_name)
  logger.setLevel(logging.INFO)
  fh = logging.FileHandler(filename, mode='a')
  fh.setLevel(logging.INFO)
  logger.addHandler(fh)
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  logger.addHandler(console)
  
  gt_list = glob.glob('/code_dafir/dataset3/gt/*.jpg')
  gt_list.sort()
#  print(gt_list)
  dir_list = sorted(os.listdir('/code_dafir/fine-tuning/test_result/')) 
#  print(len(dir_list))
#  print(dir_list)
#  record = []
  for i in range(len(dir_list)):
      if(dir_list[i]=='epoch_198' or dir_list[i]=='epoch_390' or dir_list[i]=='epoch_400'):
        print(time.localtime(time.time()))
        name = dir_list[i]
        dir_list[i] = root_path + dir_list[i]       
        pre_list = glob.glob(dir_list[i] + '/*')
        # print(i)
        print(len(pre_list))
        pre_list.sort()
        preimage_list=[]
        gtimage_list=[]
        for i in range(len(gt_list)): #36
           preImg = cv.imread(pre_list[i])
           gtImg = cv.imread(gt_list[i])
           preImg_g = cv.cvtColor(preImg, cv.COLOR_BGR2GRAY)
           gtImg_g = cv.cvtColor(gtImg, cv.COLOR_BGR2GRAY)
           preimage_list.append(preImg_g)
           gtimage_list.append(gtImg_g)
  
        ssim_now = ssim(preimage_list, gtimage_list)
        psnr_now = psnr(preimage_list, gtimage_list)
        mae_now = mae(preimage_list, gtimage_list)
        print(name + ':' + 'done')
        print("1:", ssim_now)
        print("2:", psnr_now)
        logger.info(name + ':' + '\tssim: {:.8f}\t'.format(ssim_now))
        logger.info(name + ':' + '\tpsnr: {:.8f}\t'.format(psnr_now))
        logger.info(name + ':' + '\tmae: {:.8f}\t'.format(mae_now))

  