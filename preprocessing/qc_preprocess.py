"""
Quality control for brain + SC extraction and registration preprocessing step.
See `preprocess_data.sh` for the preprocessing pipeline.
"""

import argparse
import os
from tqdm import tqdm
import imageio.v2 as imageio

import pandas as pd
import nibabel as nib
import numpy as np
import cv2

# Argument parsing
parser = argparse.ArgumentParser(description='Quality control for brain + SC extraction and registration.')
parser.add_argument('-s', '--sct_output_path', type=str, required=True,
                    help='Path to the folder generated by `sct_run_batch`. This folder should contain `data_processed` and `qc` folders.')
parser.add_argument('-mv', '--mask_voi', default=False, action='store_true',
                    help='Enable to use VOI-masked images for visualizations. Alternatively, disable to overlay the masks on images instead.')
parser.add_argument('-dv', '--disable_viz', default=False, action='store_true',
                    help='Enable to disable visualizations for this QC.')
args = parser.parse_args()

# Quick checking of arguments
if not os.path.exists(args.sct_output_path):
    raise NotADirectoryError('%s could NOT be found!' % args.sct_output_path)
else:
    if not os.path.exists(os.path.join(args.sct_output_path, 'data_processed')):
        raise NotADirectoryError('`data_processed` could NOT be found within %s' % args.sct_output_path)
    if not os.path.exists(os.path.join(args.sct_output_path, 'qc')):
        raise NotADirectoryError('`qc` could NOT be found within %s' % args.sct_output_path)

# Set environment variable for off-screen rendering with FSLeyes
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

# Get all subjects
subjects_df = pd.read_csv(os.path.join(args.sct_output_path, 'data_processed', 'participants.tsv'), sep='\t')
subjects = subjects_df['participant_id'].values.tolist()

# Perform QC for each subject
for subject in tqdm(subjects, desc='Iterating over Subjects'):
    # Get paths
    subject_images_path = os.path.join(args.sct_output_path, 'data_processed', subject)
    subject_labels_path = os.path.join(args.sct_output_path, 'data_processed', 'derivatives', 'labels', subject)

    if not os.path.exists(subject_images_path) or not os.path.exists(subject_labels_path):
        print('Could not find processed data for subject: %s' % subject)
        continue

    # (1) Check for original sessions -> does voxel dimensions match between TPs?
    ses01_fpath = os.path.join(subject_images_path, 'ses-01', 'anat', '%s_ses-01_FLAIR.nii.gz' % subject)
    ses02_fpath = os.path.join(subject_images_path, 'ses-02', 'anat', '%s_ses-02_FLAIR.nii.gz' % subject)
    ses01 = nib.load(ses01_fpath)
    ses02 = nib.load(ses02_fpath)
    if ses01.header['pixdim'].tolist() != ses02.header['pixdim'].tolist():
        print('\n\tALERT: Unmatching Voxel Dimensions found between TPs for subject: %s' % subject)

    # Set paths for files which we want to run QC on
    ses01_res_fpath = os.path.join(subject_images_path, '%s_ses-01_FLAIR_res.nii.gz' % subject)
    ses01_reg_fpath = os.path.join(subject_images_path, '%s_ses-01_FLAIR_reg-brain.nii.gz' % subject)
    ses02_res_fpath = os.path.join(subject_images_path, '%s_ses-02_FLAIR_res.nii.gz' % subject)
    mask_fpath = os.path.join(subject_images_path, 'brain_cord_mask.nii.gz')
    gtc_res_fpath = os.path.join(subject_labels_path, '%s_ses-02_FLAIR_seg-lesion_res.nii.gz' % subject)
    # NOTE: We do not check for filepaths, as `preprocess_data.sh` checks it instead before this
    # NOTE: Only including consensus GT; individual experts will be ignored for this (2) and (3)
    # NOTE: These are not the final images of sessions to be inputted to our model, but the required
    #       files to QC registration. On top of these, the final images are (i) bias-corrected,
    #       and (ii) VOI-cropped. The best way to QC this is via manual inspection in `fsleyes`

    # Load images
    ses01_res = nib.load(ses01_res_fpath)
    ses01_reg = nib.load(ses01_reg_fpath)
    ses02_res = nib.load(ses02_res_fpath)
    mask = nib.load(mask_fpath)
    gtc_res = nib.load(gtc_res_fpath)

    # Get min-max intensity values for session images (to-be-used later for viz.)
    ses01_res_min, ses01_res_max = np.min(ses01_res.get_fdata()), np.max(ses01_res.get_fdata())
    ses01_reg_min, ses01_reg_max = np.min(ses01_reg.get_fdata()), np.max(ses01_reg.get_fdata())
    ses02_res_min, ses02_res_max = np.min(ses02_res.get_fdata()), np.max(ses02_res.get_fdata())

    # (2) Basic shape checks
    if not ses01_res.shape == ses01_reg.shape == ses02_res.shape == mask.shape == gtc_res.shape:
        raise ValueError('Shape mismatch in sessions and GTs for subject: %s' % subject)

    # (3) Check if isotropic-sampling worked as expected: voxel dim should be 0.75mm x 0.75mm x 0.75mm
    if not all([x.header['pixdim'].tolist()[1:4] == [0.75, 0.75, 0.75] for x in (ses01_res, ses01_reg, ses02_res, mask, gtc_res)]):
        raise ValueError('Non-isotropic voxel dimensions observed for subject: %s' % subject)

    # (4) Check if the brain + SC mask leaves out any lesions from GTs (every expert and consensus)
    # NOTE: This is now fine as we are cropping VOI, and left-out lesions indicate mislabelled cases
    #       The ALERTs below can be discarded safely! We are leaving this here for completeness.

    gte_res_fpaths = [os.path.join(subject_labels_path, '%s_ses-02_FLAIR_lesion-manual-rater%d_res.nii.gz' % (subject, i)) for i in range(1, 5)]
    for expert_id, gt_fpath in enumerate(gte_res_fpaths + [gtc_res_fpath]):
        expert_id = 'consensus' if expert_id + 1 > 4 else str(expert_id + 1)
        gt_res = nib.load(gt_fpath)

        # Compute difference between the brain + SC mask and GT
        diff = np.ones_like(mask.get_fdata())
        diff[np.where(mask.get_fdata())] = 0
        cropped_lesions = diff * gt_res.get_fdata()

        if np.any(cropped_lesions):
            print('\n\tALERT: Lesion(s) from expert %s cropped during preprocessing for subject: %s' % (str(expert_id), subject))
            print('\t\tNumber of Voxels Cropped: %d' % np.count_nonzero(cropped_lesions))

            # Print location of the cropped voxels w.r.t. (for `fsleyes` visualization & analysis)
            voxel_coords = list(zip(*np.where(cropped_lesions != 0)))
            print('\t\tCropped Voxel Coordinates: ', voxel_coords)
            print('\t\tLESIONS OUTSIDE BRAIN + SC MASK ARE ASSUMED TO BE MISLABELLED CASES. YOU CAN SAFELY SKIP THIS ALERT.')

    if not args.disable_viz:
        # (5) Per-subject visualizations for QC on i) brain + SC extraction and ii) registration
        qc_subject_path = os.path.join(args.sct_output_path, 'qc', subject)
        if not os.path.exists(qc_subject_path):
            os.makedirs(qc_subject_path)

        # Define output paths for visualizations
        ses01_viz_fpath = os.path.join(qc_subject_path, 'ses01_res_masked.png')
        ses01_reg_viz_fpath = os.path.join(qc_subject_path, 'ses01_reg_masked.png')
        ses02_viz_fpath = os.path.join(qc_subject_path, 'ses02_res_masked.png')
        ses01_to_ses02_viz_fpath = os.path.join(qc_subject_path, 'ses01_to_ses02.gif')
        ses01_reg_to_ses02_viz_fpath = os.path.join(qc_subject_path, 'ses01_reg_to_ses02.gif')

        # Mask the volume-of-interest (VOI) if applicable
        if args.mask_voi:
            # Define output paths for output files for `fslmaths`
            ses01_res_masked_fpath = os.path.join(qc_subject_path, '%s_ses-01_FLAIR_res_qcmasked.nii.gz' % subject)
            ses01_reg_masked_fpath = os.path.join(qc_subject_path, '%s_ses-01_FLAIR_reg-brain_qcmasked.nii.gz' % subject)
            ses02_res_masked_fpath = os.path.join(qc_subject_path, '%s_ses-02_FLAIR_res_qcmasked.nii.gz' % subject)

            # Apply the brain + SC mask
            sct_mask_cmd = 'sct_maths -i %s -mul %s -o %s -v 0'
            os.system(sct_mask_cmd % (ses01_res_fpath, mask_fpath, ses01_res_masked_fpath))
            os.system(sct_mask_cmd % (ses01_reg_fpath, mask_fpath, ses01_reg_masked_fpath))
            os.system(sct_mask_cmd % (ses02_res_fpath, mask_fpath, ses02_res_masked_fpath))

            # Generate visualizations for each image file with display range as [min, max]
            fsl_render_cmd = 'fsleyes render -of %s %s -dr %0.2f %0.2f'
            os.system(fsl_render_cmd % (ses01_viz_fpath, ses01_res_masked_fpath, ses01_res_min, ses01_res_max))
            os.system(fsl_render_cmd % (ses01_reg_viz_fpath, ses01_reg_masked_fpath, ses01_reg_min, ses01_reg_max))
            os.system(fsl_render_cmd % (ses02_viz_fpath, ses02_res_masked_fpath, ses02_res_min, ses02_res_max))

            # Remove the cropped images to save space; they can easily be re-generated
            os.system('rm %s' % ses01_res_masked_fpath)
            os.system('rm %s' % ses01_reg_masked_fpath)
            os.system('rm %s' % ses02_res_masked_fpath)

        # If not applying the VOI mask, simply overlay the masks in red color for each session
        else:
            fsl_render_cmd = 'fsleyes render -of %s %s %s -dr %0.2f %0.2f -a 30 -cm red'
            os.system(fsl_render_cmd % (ses01_viz_fpath, ses01_res_fpath, mask_fpath, ses01_res_min, ses01_res_max))
            os.system(fsl_render_cmd % (ses01_reg_viz_fpath, ses01_reg_fpath, mask_fpath, ses01_reg_min, ses01_reg_max))
            os.system(fsl_render_cmd % (ses02_viz_fpath, ses02_res_fpath, mask_fpath, ses02_res_min, ses02_res_max))

        imageio.mimsave(ses01_to_ses02_viz_fpath, [imageio.imread(f) for f in (ses01_viz_fpath, ses02_viz_fpath)], duration=0.5)
        imageio.mimsave(ses01_reg_to_ses02_viz_fpath, [imageio.imread(f) for f in (ses01_reg_viz_fpath, ses02_viz_fpath)], duration=0.5)

if not args.disable_viz:
    # (6) Create aggregated visualizations for QC on i) brain + SC extraction and ii) registration
    gif_filenames = ['ses01_to_ses02.gif', 'ses01_reg_to_ses02.gif']
    subject_gif_filepaths = [[os.path.join(args.sct_output_path, 'qc', subject, f) for f in gif_filenames] for subject in subjects]
    agg_subject_gif_filepaths = [os.path.join(args.sct_output_path, 'qc', subject, '%s_agg.gif' % subject) for subject in subjects]
    agg_gif_filepath = os.path.join(args.sct_output_path, 'qc', 'agg.gif')
    # NOTE: `subject_gif_gilepaths` are populated with step (5),
    #        whereas `agg_subject_gif_filepaths` and `agg_gif_filepath` will be populated now!
    cv2_text_args = dict(fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 255), thickness=2, lineType=cv2.LINE_AA)

    for i, f in enumerate(tqdm(subject_gif_filepaths, desc='Iterating over Singular GIFs')):
        original_gif, reg_gif = imageio.get_reader(f[0]), imageio.get_reader(f[1])
        new_imgs = []

        # 1st Frame
        ses1_img = original_gif.get_next_data()
        cv2.putText(ses1_img, text='BEFORE', org=(380, 75), fontScale=1, **cv2_text_args)
        cv2.putText(ses1_img, text='ses-01', org=(380, 115), fontScale=1, **cv2_text_args)

        ses_01_reg_img = reg_gif.get_next_data()
        cv2.putText(ses_01_reg_img, text='AFTER', org=(380, 75), fontScale=1, **cv2_text_args)
        cv2.putText(ses_01_reg_img, text='ses-01_reg', org=(380, 115), fontScale=1, **cv2_text_args)
        new_imgs.append(np.hstack((ses1_img, ses_01_reg_img)))

        # 2nd Frame
        ses2_img = original_gif.get_next_data()
        cv2.putText(ses2_img, text='BEFORE', org=(380, 75), fontScale=1, **cv2_text_args)
        cv2.putText(ses2_img, text='ses-02', org=(380, 115), fontScale=1, **cv2_text_args)

        ses2_img_ = reg_gif.get_next_data()
        cv2.putText(ses2_img_, text='AFTER', org=(380, 75), fontScale=1, **cv2_text_args)
        cv2.putText(ses2_img_, text='ses-02', org=(380, 115), fontScale=1, **cv2_text_args)
        new_imgs.append(np.hstack((ses2_img, ses2_img_)))

        # Close previous GIFs and write the new GIF to file
        original_gif.close()
        reg_gif.close()
        imageio.mimsave(agg_subject_gif_filepaths[i], new_imgs, duration=0.5)

    new_imgs_frame1, new_imgs_frame2 = [], []
    for i, f in enumerate(tqdm(agg_subject_gif_filepaths, desc='Iterating over Aggregated GIFs')):
        agg_subject_gif = imageio.get_reader(f)

        # 1st Frame
        img1 = agg_subject_gif.get_next_data()
        cv2.putText(img1, text=subjects[i], org=(675, 75), fontScale=2, **cv2_text_args)
        new_imgs_frame1.append(img1)

        # 2nd Frame
        img2 = agg_subject_gif.get_next_data()
        cv2.putText(img2, text=subjects[i], org=(675, 75), fontScale=2, **cv2_text_args)
        new_imgs_frame2.append(img2)

        # Close previous aggregated subject GIFs
        agg_subject_gif.close()

    # Concatenate all of the aggregated subject GIFs into a single GIF
    # NOTE: To concatenate properly, we need matching dims; use min height and width amongst all!
    min_h = min([img.shape[0] for img in new_imgs_frame1])
    min_w = min([img.shape[1] for img in new_imgs_frame2])
    new_imgs_frame1 = np.concatenate([img[:min_h, :min_w, :] for img in new_imgs_frame1], axis=0)
    new_imgs_frame2 = np.concatenate([img[:min_h, :min_w, :] for img in new_imgs_frame2], axis=0)
    imageio.mimsave(agg_gif_filepath, [new_imgs_frame1, new_imgs_frame2], duration=0.5)
