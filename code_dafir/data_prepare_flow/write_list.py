import glob  
import random

img_list = glob.glob('./picture/data_256/*/*/*.jpg') + glob.glob('./picture/data_256/*/*/*/*.jpg')   #全局路径中寻找，*代表通配符，三层列表的图片路径和四层列表的图片路径收集起来，返回一个列表
random.shuffle(img_list)
print(len(img_list))  

file = open('img_path.txt', "w")
file.write(str(img_list))
file.close()
