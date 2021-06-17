# `modeling`

This folder includes custom modeling efforts. Currently, we include implementations for TransUNet3D. 
Check [#44](https://github.com/ivadomed/ms-challenge-2021/pull/44) to learn more.


## How to Run
Model-parallel training can be started respectively with the following command:
```
export CUDA_VISIBLE_DEVICES=<[GPU_IDs]>
python train.py -id <MODEL_ID> -dr ~/duke/projects/ivadomed/tmp_ms_challenge_2021_preprocessed/ -loc 0 -bs 8 -nw 0
```

Data-parallel training can be started respectively with the following command:
```
export CUDA_VISIBLE_DEVICES=<[GPU_IDs]>
python train.py -id <MODEL_ID> -dr ~/duke/projects/ivadomed/tmp_ms_challenge_2021_preprocessed/ -loc -1 -bs 32 -nw 8
```