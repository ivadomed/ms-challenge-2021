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

### Prerequisites

You need 
* [`SCT`](https://spinalcordtoolbox.com/en/latest/), 
* `ANIMA`, and 
* [`FSL`](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) 

installed for the preprocessing pipeline.

Check [this](https://github.com/ivadomed/ms-challenge-2021/issues/4#issuecomment-849039537) to
see how `ANIMA` should be installed and configured in your system.

## Dataset Curation
We used the following script to curate the dataset and make it compatible with the Brain Imaging Data Structure (BIDS) specification:
```
python scripts/curate_msseg_challenge_2021.py -d PATH_TO_msseg_challenge_2021_DATASET
```

## Preprocessing
The preprocessing pipeline for each subject can be found in `preprocessing/preprocess_data.sh`. 
The quality control (QC) script for this pipeline is `preprocessing/qc_preprocess.py`.
Section 3.2 of our arXiv paper describes this pipeline in detail. 
You can also check our [quality-control (QC) visualizations](https://github.com/ivadomed/ms-challenge-2021/releases/download/v0.1/qc_registration.gif). 
You can find an example preprocessing visualization for a subject below.

<p align="center">
  <img src="https://github.com/ivadomed/ms-challenge-2021/releases/download/v0.1/qc_registration_sub99.gif" width=800 />
</p>

### Preprocessing steps
The preprocessing steps include:
1. Resampling of both FLAIR sessions to isotropic 0.75mm x 0.75mm x 0.75mm resolution
2. Spinal cord (SC) segmentation with `sct_deepseg_sc` on both sessions
3. Initial registration (`ses-01 -> ses-02`) using `sct_register_multimodal` with the help of SC segmentation masks
4. Brain extraction using `bet2` on second session
5. Finer registration (`ses-01 -> ses-02`) using `antsRegistration`
6. Brain + SC masking on both sessions
7. Bias-correction using `animaN4BiasCorrection` on both sessions
8. Cropping of volume-of-interest (VOI)

### How to run preprocessing

We are using `sct_run_batch` to perform preprocessing on all subjects:
```
sct_run_batch -path-data PATH_DATA -path-output PATH_OUTPUT -script preprocessing/preprocess_data.sh -script-args "bet2"
```
where `PATH_DATA` is the path to the BIDS data folder, and `PATH_OUTPUT` is where the output of
preprocessing will be saved to. `PATH_OUTPUT` will contain `data_processed` and `qc` 
(among others) directories after a successful run.

Additionally, you might want to play around with `-jobs` and `-itk-threads` arguments of 
`sct_run_batch` to gain speed-ups. `-jobs 16 -itk-threads 16` was what we used in `joplin`.

After a successful run, next step is to do quality-control (QC):
```
python preprocessing/qc_preprocess.py -s PATH_OUTPUT -mv
```

The QC script checks for:
* whether resolutions match between the two original sessions,
* whether all image sizes are equivalent for each subject,
* whether isotropic-resampling (step 1.) worked as expected,
* whether brain + SC mask leaves out any lesions from GTs (every expert and consensus),
    * *NOTE*: In this project, we assumed that any lesions that are not inside the brain + SC region
      reflects a mistake in the annotation process. Therefore, you can discard this check safely!

and outputs the following to QC i) brain + SC extraction and ii) registration:
* per-subject visualizations 
* aggregated visualizations (all subjects)

## Modeling
All modeling efforts can be found in `modeling/`. `modeling/README.md` and section 3.4 of our arXiv paper describe the deep learning architectures used in detail.

## Internal use at NeuroPoly

Location of the data:
- `git+ssh://data.neuro.polymtl.ca:msseg_challenge_2021` --> this is the BIDS-converted dataset.
