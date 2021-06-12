# `preprocessing`

This folder handles preprocessing of patient FLAIR volumes before the modelling component. The
preprocessing pipeline for each subject can be found in `preprocess_data_sh.` The quality control 
(QC) script for this pipeline is `qc_preprocess.py`.

## Steps
The preprocessing steps include:
1. Resampling of both FLAIR sessions to isotropic 0.5mm x 0.5mm x 0.5mm resolution
2. Spinal cord (SC) segmentation with `sct_deepseg_sc` on both sessions
3. Initial registration (`ses-01 -> ses-02`) using `sct_register_multimodal` with the help of SC segmentation masks
4. Brain extraction using `bet2` on second session
5. Finer registration (`ses-01 -> ses-02`) using `antsRegistration`
6. Brain + SC masking on both sessions
7. Bias-correction using `animaN4BiasCorrection` on both sessions
8. Cropping of volume-of-interest (VOI)

## Prerequisites

You need [`SCT`](https://spinalcordtoolbox.com/en/latest/), `ANIMA`, and [`FSL`](https://fsl.fmrib.ox.ac.uk/fsl/fslwiki/) installed.

Check [this](https://github.com/ivadomed/ms-challenge-2021/issues/4#issuecomment-849039537) to
see how `ANIMA` should be installed and configured in your system.


## How to Run

We are using `sct_run_batch` to perform preprocessing on all subjects:
```
sct_run_batch -path-data PATH_DATA -path-output PATH_OUTPUT -script preprocess_data.sh
```
where `PATH_DATA` is the path to the BIDS data folder, and `PATH_OUTPUT` is where the output of
preprocessing will be saved to. `PATH_OUTPUT` will contain `data_processed` and `qc` 
(among others) directories after a successful run.

Additionally, you might want to play around with `-jobs` and `-itk-threads` arguments of 
`sct_run_batch` to gain speed-ups. `-jobs 16 -itk-threads 16` was what we used in `joplin`.

After a successful run, next step is to do quality-control (QC):
```
python qc_preprocessing.py -s PATH_OUTPUT -mv
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