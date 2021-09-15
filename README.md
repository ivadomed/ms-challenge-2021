# Team Neuropoly: MICCAI 2021 MS New Lesions Segmentation Challenge
This repository contains implementations of segmentation pipelines proposed by the NeuroPoly lab for the MICCAI 2021 MS New Lesions Segmentation Challenge. The goal of the challenge was to segment the new lesions given two fluid attenuated inversion recovery (FLAIR) images (baseline and follow-up). You can also check our [arXiv paper](https://arxiv.org/pdf/2109.05409.pdf).

<p align="center">
  <img src="https://github.com/ivadomed/ms-challenge-2021/releases/download/v0.1/main.gif" width=300 />
</p>

## Dataset Curation
We used the following script to curate the dataset and make it compatible with the Brain Imaging Data Structure (BIDS) specification:
```
python scripts/curate_msseg_challenge_2021.py -d PATH_TO_msseg_challenge_2021_DATASET
```

## Preprocessing
The preprocessing pipeline for each subject can be found in `/preprocessing`. `preprocessing/README.md` and section 3.2 of our arXiv paper describes this pipeline in detail. You can also check our [quality-control (QC) visualizations](https://github.com/ivadomed/ms-challenge-2021/releases/download/v0.1/qc_registration.gif).

## Modeling
All modeling efforts can be found in `/modeling`. `modeing/README.md` and section 3.4 of our arXiv paper describes the deep learning arhictectures used in detail.
