# DaFIR: Distortion-aware Representation Learning for Fisheye Image Rectification

![1683021188130](https://user-images.githubusercontent.com/91788329/235635829-b7536568-6723-4059-9ffd-56a6e3ee7839.png)

## Greetings
     Dear researchers and engineers, good afternoon. Due to my busy work and study schedule, the DaFIR project has only been fully open-sourced today. Thank you for your patience. 
Below is a detailed description of this project.

## Inference 
1. The pretrained model can be download from [Baidu Cloud](https://pan.baidu.com/s/1J97k1TSNyMicowRLZ7KJvw?pwd=ecmf)(Extraction code: ecmf). Put the model to `$ROOT/test/save/net/`.
2. Put the distorted images in `$ROOT/dataset/data/`.
3. Distortion rectification. The rectified images are saved in `$ROOT/test/test_result/` by default.
    ```
    python test.py
    ```

## Acknowledgement
The codes are largely based on [PCN](https://github.com/uof1745-cmd/PCN) and [MAE](https://github.com/facebookresearch/mae). Thanks for their wonderful works.
