# DaFIR
<p>
    <a href='https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10251977' target="_blank"><img src="https://img.shields.io/badge/Paper-IEEE-blue"></a>
</p>

![1](https://user-images.githubusercontent.com/91788329/235635829-b7536568-6723-4059-9ffd-56a6e3ee7839.png)

> [DaFIR: Distortion-Aware Representation Learning for Fisheye Image Rectification](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10251977)  
> TCSVT 2023

Any questions or discussions are welcomed!

## Contact details updated
Since I have graduated from USTC. Please contact me on a new email address for faster response：liaozhaokang.lzk@antgroup.com

## Greetings
1. Dear researchers and engineers, good afternoon. Due to my busy work and study schedule, the DaFIR project has only been fully open-sourced today. Thank you for your patience. 
Below is a detailed description of this project.

## Dependencies
1. python 3.8
2. cudnn 8.2.1
3. pytorch 1.9.0
4. cuda 11.2
5. numpy 1.22.3
6. skimage 0.20.0
7. opencv 4.7.0
8. Pillow 9.4.0
9. timm 0.6.13

## Source Data
1. In this project, we synthesize a fisheye image from a group of distortion parameters and a source image without distortion.
2. The source images in the size of 256*256 can be download from [Baidu Cloud](https://pan.baidu.com/s/1M2653RTWun1vDaj1BFPF2A?pwd=gjq3) (Extraction code: gjq3 ). Download the file "picture.zip",
extract to the fold "picture" and put it into the path /code_dafir/data_prepare/. 
3. The path list of the above source images can be download from [Baidu Cloud](https://pan.baidu.com/s/1y1YEH4NZK51KjOfpJB8TMw?pwd=9p97) (Extraction code: 9p97 ). Download the file "img_path.txt"
and put it into the path /code_dafir/data_prepare_ddm/ and /code_dafir/data_prepare_flow/. For this project, we use the index 1-200K in the "img_path.txt" for pre-training and index 200k-300k in the img_path.txt for fine-tuning and the index 300,000 to 305,000 for testing.

## Synthesize Fisheye Images and Labels
1. Synthesize Fisheye Images and DDM (D labels) for pretraining dataset
   ```
   cd /code_dafir/data_prepare_ddm/
   python get_dataset_ddm.py --index 0
   ```
   The index parameter can be adjust to change different source images for synthesization
   
2. Synthesize Fisheye Images and Pixel-wise Flow Maps for fine-tuning dataset 
   ```
   cd /code_dafir/data_prepare_flow/
   python get_dataset.py --index 0
   ```
   The index parameter can be adjust to change different source images for synthesization

## Pretraining
1. Input Fisheye Images and Predict D Labels.
   ```
   cd /code_dafir/pre_training/
   python -m torch.distributed.launch --nproc_per_node=2 --master_port 1285 main.py
   ```
   The parameter --master_port can be selected randomly. Use 2 GPUs for distributed trainning.
2. After Pretraining, the model is saved in /code_dafir/pre_training/save/net/

## Fine-tuning
1. Put a Pre-traning release model into the path /code_dafir/fine-tuning/pretrain/
2. Input Fisheye Images and Predict Pixel-wise Flow Map.
   ```
   cd /code_dafir/fine-tuning/
   python -m torch.distributed.launch --nproc_per_node=2 --master_port 1285 main.py
   ```
   The parameter --master_port can be selected randomly. Use 2 GPUs for distributed trainning.
3. After fine-tuning, the model is saved in /code_dafir/fine-tuning/save/net/

## Testing
1. We provide a test dataset, which is the same with the one in our paper (https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10251977). This dataset can be download from
   [Baidu Cloud] (https://pan.baidu.com/s/1xXuCdmgrjGlwQeDIeCMFgQ?pwd=yzv7)(Extraction Code: yzv7). Unrap the fold dataset3 and put it into the path /code_dafir/.
2. Testing the model in synthesize fisheye images
   ```
   cd /code_dafir/fine-tuning/
   python test.py
   ```
3. Testing the model in real fisheye images
   ```
   cd /code_dafir/fine-tuning/
   python test2.py
   ```

## Citation

If you find this code useful for your research, please use the following BibTeX entry.

  ```
  @article{liao2023dafir,
    title={DaFIR: Distortion-aware Representation Learning for Fisheye Image Rectification},
    author={Liao, Zhaokang and Zhou, Wengang and Li, Houqiang},
    journal={IEEE Transactions on Circuits and Systems for Video Technology},
    year={2023},
    publisher={IEEE}
  }
  ```

     
## Evaluation 
1. Evaluate the model with metric PSNR and SSIM
   ```
   cd /code_dafir/core/
   python compare.py
   ```

## Contact us 
  Any questions please contact me in lzk9508@mail.ustc.edu.cn. 

## Acknowledgement
The codes are largely based on [PCN](https://github.com/uof1745-cmd/PCN) and [MAE](https://github.com/facebookresearch/mae). Thanks for their wonderful works.
