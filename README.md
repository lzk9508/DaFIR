# DaFIR: Distortion-aware Representation Learning for Fisheye Image Rectification

![1683018123723](https://user-images.githubusercontent.com/91788329/235624953-a1f08090-7d9c-404d-ae27-30d80d328e2b.png)

## Inference 
1. The pretrained model can be download from [Baidu Cloud](https://pan.baidu.com/s/1J97k1TSNyMicowRLZ7KJvw?pwd=ecmf)(Extraction code: ecmf). Put the model to `$ROOT/test/save/net/`.
2. Put the distorted images in `$ROOT/dataset/data/`.
3. Distortion rectification. The rectified images are saved in `$ROOT/test/test_result/` by default.
    ```
    python test.py
    ```

## Acknowledgement
The codes are largely based on [PCN](https://github.com/uof1745-cmd/PCN) and [MAE](https://github.com/facebookresearch/mae). Thanks for their wonderful works.
