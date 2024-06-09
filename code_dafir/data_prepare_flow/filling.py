import numpy as np
import cv2 as cv
import math

HEIGHT = 400
WIDTH = 400
break_pix = 500
size = 3

def getpix(cut):
    cor = []
    value = []
    for i in range(0, size + size):     #cut窗口的尺寸，两倍size。
        for j in range(0, size + size):
            if(cut[i, j] != break_pix):
                cor.append([i,j])
                value.append(cut[i,j])
    return cor, value

def getnear(cors, values, now_cor):
    min = 100
    value = break_pix
    for i in range(0, len(cors)):
        cor = cors[i]
        leng = math.sqrt(pow(cor[0] - now_cor[0], 2) + pow(cor[1] - now_cor[1], 2))
        if(leng < min):
            min = leng
            value = values[i]
    return value

def compensate(srcImg):    #对最终比例尺下畸变图与原图差值的补偿。这个图的外立面是500，即边界外是500
    dstImg = np.zeros([HEIGHT, WIDTH], np.float32)
    srcImg = cv.copyMakeBorder(srcImg, size, size, size, size, cv.BORDER_CONSTANT, value=0)  #对输入图像四个方向上各补三个零
    for i in range(size, HEIGHT + size):
        for j in range(size, WIDTH + size):
            if (srcImg[i, j] != break_pix):    #原补偿量图不超过500
                dstImg[i - size, j - size] = srcImg[i, j]   #直接读数过来。 
            else:
                if(i < 3 * size or i > WIDTH - (2 * size)):
                    if(i < 3 * size):  #3区 
                        w_start = size
                        w_end = 3 * size
                        now_x = i - size
                    else:            #2区
                        w_start = WIDTH - (2 * size)
                        w_end = WIDTH
                        now_x = i - (WIDTH - (2 * size))
                else:                #1区
                    w_start = i - size
                    w_end = i + size
                    now_x = size

                if (j < 3 * size or j > HEIGHT - (2 * size)):  
                    if (j < 3 * size):
                        h_start = size
                        h_end = 3 * size
                        now_y = j - size
                    else:
                        h_start = WIDTH - (2 * size)
                        h_end = WIDTH
                        now_y = j - (WIDTH - (2 * size))
                else:
                    h_start = j - size
                    h_end = j + size
                    now_y = size

                cut = srcImg[w_start:w_end, h_start:h_end]    #对于每个500的外点，把这个样的窗口剪切出来。 
                # cut = srcImg[i - size:i + size, j - size:j + size]
                # print(cut)
                cors, value = getpix(cut)   #把窗口范围内内点的坐标和像素值提取出来。
                rgb = getnear(cors, value, [now_x, now_y])   #now_x,now_y表示当前点在cut点列表中的排序位置。
                dstImg[i - size, j - size] = rgb             #cut点列表中选择离当前点最近的值，超出最近值为默认500. 
    return dstImg