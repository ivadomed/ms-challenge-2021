# Team Neuropoly: MICCAI 2021 MS New Lesions Segmentation Challenge
This repository contains implementations of segmentation pipelines proposed by the NeuroPoly lab for the MICCAI 2021 MS New Lesions Segmentation Challenge. The goal of the challenge was to segment the new lesions given two fluid attenuated inversion recovery (FLAIR) images (baseline and follow-up). You can also check our [arXiv paper](https://arxiv.org/pdf/2109.05409.pdf) and [poster](https://portal.fli-iam.irisa.fr/files/2021/09/MSSEG2_Poster_Team15.pdf).

<p align="center">
  <img src="https://github.com/ivadomed/ms-challenge-2021/releases/download/v0.1/main.gif" width=300 />
</p>

## Getting Started
This repo has been tested with Python 3.8. Follow the steps below to use this repo:
1. Clone project: `git clone https://github.com/ivadomed/ms-challenge-2021`
2. Create virtual environment and install packages:
	```
	cd ms-challenge-2021/
	virtualenv -p python3.8 .env
	source .env/bin/activate
	pip install -r requirements.txt
	```
3. Check the sections below for dataset curation, preprocessing, and modeling. `modeling/README.md` provides documentation on how to run training and evaluation using the codebase.

## Dataset Curation
We used the following script to curate the dataset and make it compatible with the Brain Imaging Data Structure (BIDS) specification:
```
python scripts/curate_msseg_challenge_2021.py -d PATH_TO_msseg_challenge_2021_DATASET
```

## Preprocessing
The preprocessing pipeline for each subject can be found in `preprocessing/`. `preprocessing/README.md` and section 3.2 of our arXiv paper describes this pipeline in detail. You can also check our [quality-control (QC) visualizations](https://github.com/ivadomed/ms-challenge-2021/releases/download/v0.1/qc_registration.gif). You can find an example preprocessing visualization for a subject below.

<p align="center">
  <img src="https://github.com/ivadomed/ms-challenge-2021/releases/download/v0.1/qc_registration_sub99.gif" width=800 />
</p>

## Modeling
All modeling efforts can be found in `modeling/`. `modeling/README.md` and section 3.4 of our arXiv paper describe the deep learning architectures used in detail.
