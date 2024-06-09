# coding=utf-8
import cv2 as cv 
import numpy as np
import random
import os
from filling import compensate
from cut import get_new_gt2, collect_edges, get_flow_gt
import glob
import argparse

HEIGHT = 400
WIDTH = 400
final_lon = 256


def create_fish(srcImg, k1, k2, k3, k4):
    up = []
    down = []
    left = []
    right = []
    
    dstImg = np.zeros([HEIGHT, WIDTH, 3], np.uint8) + 255
    det_uv = np.zeros([HEIGHT, WIDTH, 2], np.int32) + 500
    ddm_Img = np.zeros([256, 256, 1], np.float32)

    x0 = (WIDTH - 1)/ 2.
    y0 = (HEIGHT - 1) / 2.
    
    x1 = 127.5
    y1 = 127.5

    min_x = 9999
    cut_r = 0
    
    for i in range(0, HEIGHT):
        for j in range(0, WIDTH):
            x_d = j - x0
            y_d = i - y0
            rd_2 = pow(x_d, 2) + pow(y_d, 2)
            rd_4 = pow(rd_2, 2)
            rd_6 = pow(rd_2, 3)
            rd_8 = pow(rd_2, 4)
            x = (1 + k1 * rd_2 + k2 * rd_4 + k3 * rd_6 + k4 * rd_8) * x_d
            y = (1 + k1 * rd_2 + k2 * rd_4 + k3 * rd_6 + k4 * rd_8) * y_d
            if (int(y) == int(-y0) and x >= -x0 and x <= x0):
                if (x < min_x):
                    min_x = x
                    cut_r = -x_d   

    start = int(x0 - cut_r)
    end = int(x0 + cut_r) + 1
    
    for i in range(start, end):
        for j in range(start, end):
            x_d = j - x0
            y_d = i - y0
            rd_2 = pow(x_d, 2) + pow(y_d, 2)
            rd_4 = pow(rd_2, 2)
            rd_6 = pow(rd_2, 3)
            rd_8 = pow(rd_2, 4)
            x = (1 + k1 * rd_2 + k2 * rd_4 + k3 * rd_6 + k4 * rd_8) * x_d
            y = (1 + k1 * rd_2 + k2 * rd_4 + k3 * rd_6 + k4 * rd_8) * y_d
            u = int(x + x0)
            v = int(y + y0)
            if (u >= 0 and u < WIDTH) and (v >= 0 and v < HEIGHT):
                dstImg[i, j, 0] = srcImg[v, u, 0]
                dstImg[i, j, 1] = srcImg[v, u, 1]
                dstImg[i, j, 2] = srcImg[v, u, 2]
                up, down, left, right = collect_edges(start, end, j, i, u, v, up, down, left, right)
                cut_r_gt = int(x0 - up[0][0])
                parameter_c = float(cut_r_gt) / float(cut_r)
                parameter_b = float(final_lon) / float(cut_r_gt*2)
                det_uv[v, u, 0] = (((parameter_c * x_d) + x0) - u) * parameter_b
                det_uv[v, u, 1] = (((parameter_c * y_d) + y0) - v) * parameter_b


    cropImg = dstImg[(int(x0) - int(cut_r)):(int(x0) + int(cut_r)), (int(y0) - int(cut_r)):(int(y0) + int(cut_r))]
    dstImg2 = cv.resize(cropImg, (final_lon, final_lon), interpolation=cv.INTER_LINEAR)
    for i in range(0,256):
       for j in range (0,256):
            x_d = i - x1
            y_d = j - y1
            rd_2 = pow(x_d, 2) + pow(y_d, 2)
            rd_4 = pow(rd_2, 2)
            rd_6 = pow(rd_2, 3)
            rd_8 = pow(rd_2, 4)
            ddm_Img[i,j,0] = (1 + k1 * rd_2 + k2 * rd_4 + k3 * rd_6 + k4 * rd_8)
            if ddm_Img[i,j,0] * pow(rd_2, .5) > 127.5:   #only calculate the DDM values for the insribed circle region
                ddm_Img[i,j,0] = 0
        
    return dstImg2, ddm_Img                            

path = 'picture/'
k1_top = 1e-4
k1_down = 1e-6

k2_top = 1e-9
k2_down = 1e-11

k3_top = 1e-14
k3_down = 1e-16

k4_top = 1e-19
k4_down = 1e-21

num = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0)   
    args = parser.parse_args()
    
    start = args.index * 10000
    img_list = eval(open('img_path.txt', 'r').read())[start : start + 10000]
    
    print(start)
    print(len(img_list))
    
    file = open('./para' + str(args.index) + '.txt', "w")
    
    for files in img_list:
    
        if(num % 100 == 0):
            print(num)
            
        k1 = -random.uniform(k1_down, k1_top)
        k2 = -random.uniform(k2_down, k2_top)
        k3 = -random.uniform(k3_down, k3_top)
        k4 = -random.uniform(k4_down, k4_top)
        
        srcImg = cv.imread(files)
        
        source = cv.resize(srcImg, (256, 256))
        
        parameters=[]
        parameters.append(k1)
        parameters.append(k2)
        parameters.append(k3)
        parameters.append(k4)

        file.write(str(start+num).zfill(8) + str(parameters) + "\n")
        
        srcImg = cv.resize(srcImg, (400, 400))
        cutImg, ddm_Img = create_fish(srcImg, -k1, -k2, -k3, -k4)
        cv.imwrite('../dataset1/data/' + str(start+num).zfill(6) + '.jpg', cutImg)
        np.save('../dataset1/ddm/' + str(start+num).zfill(6) + '.npy', ddm_Img)
        num = num + 1 
    
    file.close()
    
