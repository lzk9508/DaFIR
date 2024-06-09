# coding=utf-8
import cv2 as cv                                      
import numpy as np
import random
import os
from filling import compensate
from cut import collect_edges, get_flow_gt
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
    flowImg2 = np.zeros([HEIGHT, WIDTH, 2], np.float32)                      #pixel wise flow map
     
    x0 = (WIDTH - 1)/ 2.
    y0 = (HEIGHT - 1) / 2.

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

    r_d = 0 
   
    cropImg = dstImg[(int(x0) - int(cut_r)):(int(x0) + int(cut_r)), (int(y0) - int(cut_r)):(int(y0) + int(cut_r))]  
    dstImg2 = cv.resize(cropImg, (final_lon, final_lon), interpolation=cv.INTER_LINEAR)         
    
              
    for u in range(0, 400):
        for v in range(0, 400): #
             x = u - x0
             y = v - y0
             args = [k4, 0, k3, 0, k2, 0, k1, 0, 1, -pow(pow(x,2)+pow(y,2),0.5)]
             r_root = np.roots(args)
             r_d = r_root[8].real
             rd_2 = pow(r_d, 2)
             rd_4 = pow(rd_2, 2)
             rd_6 = pow(rd_2, 3)
             rd_8 = pow(rd_2, 4)
             x_d = x/(1 + k1 * rd_2 + k2 * rd_4 + k3 * rd_6 + k4 * rd_8)
             y_d = y/(1 + k1 * rd_2 + k2 * rd_4 + k3 * rd_6 + k4 * rd_8)
             i = x_d+128.0
             j = y_d+128.0
             flowImg2[u, v, 0] = i
             flowImg2[u, v, 1] = j
             
    a = flowImg2[393,393,0]-flowImg2[5,5,0]
    a = a.astype('float64')
    a = 255.0/a        
    flowImg2 = cv.resize(flowImg2, (final_lon, final_lon), interpolation=cv.INTER_LINEAR)
    flowImg2 = (flowImg2 - 128.0) * a + 127.7
    return dstImg, dstImg2, flowImg2                                      


k1_top = -4.0    
k1_down = -6.0

k2_top = -9.0   
k2_down = -11.0

k3_top = -14.0   
k3_down = -16.0

k4_top = -19.0   
k4_down = -21.0 

num = 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--index', type=int, default=0) 
    args = parser.parse_args() 


    start = args.index*1000
    img_list = eval(open('img_path.txt', 'r').read())[args.index*1000:(args.index)*1000+25000]
    print(len(img_list))

    file = open('../dataset2/para' + str(args.index) + '.txt', "w")                  
    
    for files in img_list:
        e1 = random.uniform(k1_down, k1_top)
        e2 = random.uniform(k2_down, k2_top)
        e3 = random.uniform(k3_down, k3_top)
        e4 = random.uniform(k4_down, k4_top)
        k1 = -pow(10, e1)
        k2 = -pow(10, e2)
        k3 = -pow(10, e3)
        k4 = -pow(10, e4)
        
        if( num % 100 == 0 ):
            print(files)
        srcImg = cv.imread(files)
        
        source = srcImg
        parameters=[]
        parameters.append(k1)
        parameters.append(k2)
        parameters.append(k3)
        parameters.append(k4)

        file.write(str(start+num).zfill(8) + str(parameters) + "\n")

        srcImg = cv.resize(srcImg, (400, 400))
        dstImg, cutImg, flowImg2 = create_fish(srcImg, -k1, -k2, -k3, -k4)

        cv.imwrite('../dataset2/data/' + str(start+num).zfill(6) + '.jpg', cutImg)
        np.save('../dataset2/flow/' + str(start+num).zfill(6) + '.npy', flowImg2)
        num = num + 1 
        if( num % 100 == 1 ):
            print('done: ', files)
   
    file.close()                                   
    
