import cv2 as cv
import glob
import os
import random

if __name__ == "__main__":
    img_list = eval(open('img_path.txt', 'r').read())
    random.shuffle(img_list)
    file = open('img_path.txt', "w")
    file.write(str(img_list))
    file.close()