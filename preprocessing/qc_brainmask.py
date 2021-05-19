import os
import nibabel as nib
import numpy as np
import argparse

# Argument parsing
parser = argparse.ArgumentParser(
    description="""Preprocess data for the Longitudinal MS Lesion Segmentation Challenge of MICCAI 2021 with the anima library. 
                    The preprocessing consists in a brain extraction followed by a bias field correction.""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-s', '--source', type=str, required=True, help="""Input folder containing the data before ANIMA preprocessing.
The folder must follow this structure:
/input/folder/
 013
    brain_mask.nii.gz
    flair_time01_on_middle_space.nii.gz
    flair_time02_on_middle_space.nii.gz
    ground_truth_expert1.nii.gz
    ground_truth_expert2.nii.gz
    ground_truth_expert3.nii.gz
    ground_truth_expert4.nii.gz
    ground_truth.nii.gz
 015
    brain_mask.nii.gz
    flair_time01_on_middle_space.nii.gz
    flair_time02_on_middle_space.nii.gz
    ground_truth_expert1.nii.gz
    ground_truth_expert2.nii.gz
    ground_truth_expert3.nii.gz
    ground_truth_expert4.nii.gz
    ground_truth.nii.gz
...
""")

parser.add_argument('-d', '--dest', type=str, required=True, help="""Input folder containing th data after ANIMA preprocessing.
The folder must follow this structure:
/input/folder/
 013
    brain_mask.nii.gz
    flair_time01_on_middle_space.nii.gz
    flair_time02_on_middle_space.nii.gz
    ground_truth_expert1.nii.gz
    ground_truth_expert2.nii.gz
    ground_truth_expert3.nii.gz
    ground_truth_expert4.nii.gz
    ground_truth.nii.gz
 015
    flair_time01_on_middle_space.nii.gz
    flair_time02_on_middle_space.nii.gz
    ground_truth_expert1.nii.gz
    ground_truth_expert2.nii.gz
    ground_truth_expert3.nii.gz
    ground_truth_expert4.nii.gz
    ground_truth.nii.gz
...
""")

args = parser.parse_args()

src_folder = args.source
dest_folder = args.dest

flairs = ['flair_time01_on_middle_space.nii.gz', 'flair_time02_on_middle_space.nii.gz']
mask = ['brain_mask.nii.gz']

for patientName in os.listdir(src_folder):

    src_patient_folder = os.path.join(src_folder, patientName)
    dest_patient_folder = os.path.join(dest_folder, patientName)

    if not os.path.isdir(src_patient_folder): continue
    if not os.path.isdir(dest_patient_folder): continue

    print("Checking brain extraction of patient " + patientName + "...")

    for flairName in flairs:
        src_flair = os.path.join(src_patient_folder, flairName)
        mask_patient = os.path.join(dest_patient_folder, mask)
        # Output fname
        output_fname = os.path.join(dest_patient_folder, 'difference_'+flairName.split('.nii.gz')[0]+'.nii.gz')

        # Read data
        src_im = nib.load(src_flair)
        mask_im = nib.load(mask_patient)

        # Compute difference between the raw data and the brain mask
        diff_np = np.copy(src_im.data)
        diff_np[np.where(mask_im)] = 0

        # Save difference
        diff_nib = nib.Nifti1Image(dataobj=diff_np, affine=None, header=src_im.header.copy())
        nib.save(diff_nib, output_fname)
        print('\tSaving difference in {}'.format(output_fname))
