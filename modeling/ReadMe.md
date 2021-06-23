### Probabilistic U-Net for Longitudinal MS Lesion Segmentation
This contains the first attempt of using PU-Net for segmenting MS Lesions.

The current (tested) version supports single-GPU training. How to run:

```
$ export CUDA_VISIBLE_DEVICES=<your-GPU-id-here>
$ python training.py -fd 1.0
```
