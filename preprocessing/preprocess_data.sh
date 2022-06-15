#!/bin/bash
#
# Preprocess data.
#
# Dependencies:
# - FSL (includes bet2): 5.0.11
# - SCT: 5.3.0
# - ANIMA (includes animaBrainExtraction): 4.0.1
# - ANTs: 2.2.0
#
# Usage:
#   ./preprocess_data.sh <SUBJECT> <BRAIN_EXTRACTION_METHOD>
#
# Brain extraction method: Use 'anima' to call animaBrainExtraction, otherwise bet2 is used.
#
# Manual segmentations or labels should be located under:
# PATH_DATA/derivatives/labels/SUBJECT/<CONTRAST>/

# The following global variables are retrieved from the caller sct_run_batch
# but could be overwritten by uncommenting the lines below:
# PATH_DATA_PROCESSED="~/data_processed"
# PATH_RESULTS="~/results"
# PATH_LOG="~/log"
# PATH_QC="~/qc"

# Uncomment for full verbose
set -x

# Immediately exit if error
set -e -o pipefail

# Exit if user presses CTRL+C (Linux) or CMD+C (OSX)
trap "echo Caught Keyboard Interrupt within script. Exiting now.; exit" INT

# Retrieve input params and other params
SUBJECT=$1
BRAIN_EXTRACTION_METHOD=$2
ANIMA_SCRIPTS_PUBLIC_PATH=$(grep "^anima-scripts-public-root = " ~/.anima/config.txt | sed 's/.* = //')
ANIMA_BINARIES_PATH=$(grep "^anima = " ~/.anima/config.txt | sed 's/.* = //')

# get starting time:
start=`date +%s`


# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED

# Copy BIDS-required files to processed data folder (e.g. list of participants)
if [[ ! -f "participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv .
fi
if [[ ! -f "participants.json" ]]; then
  rsync -avzh $PATH_DATA/participants.json .
fi
if [[ ! -f "dataset_description.json" ]]; then
  rsync -avzh $PATH_DATA/dataset_description.json .
fi
if [[ ! -f "README" ]]; then
  rsync -avzh $PATH_DATA/README .
fi

# Copy source images
rsync -avzh $PATH_DATA/$SUBJECT .

# Copy segmentation GTs
mkdir -p derivatives/labels
rsync -avzh $PATH_DATA/derivatives/labels/$SUBJECT derivatives/labels/.

# (1) Go to subject folder for source images
cd ${SUBJECT}

# TODO: re-think how file variable is defined-- not clean to have folders in there
file_ses1_onlyfile="${SUBJECT}_ses-01_FLAIR"
file_ses1="ses-01/anat/${SUBJECT}_ses-01_FLAIR"
file_ses2_onlyfile="${SUBJECT}_ses-02_FLAIR"
file_ses2="ses-02/anat/${SUBJECT}_ses-02_FLAIR"

# Resample both sessions to isotropic 0.75mm x 0.75mm x 0.75mm voxel dimensions
flirt -in ${file_ses1}.nii.gz -ref ${file_ses1}.nii.gz -applyisoxfm 0.75 -nosearch -out ${file_ses1_onlyfile}_res.nii.gz
flirt -in ${file_ses2}.nii.gz -ref ${file_ses2}.nii.gz -applyisoxfm 0.75 -nosearch -out ${file_ses2_onlyfile}_res.nii.gz

# Spinal cord extraction
sct_deepseg_sc -i ${file_ses1_onlyfile}_res.nii.gz -c t1 -o ${file_ses1_onlyfile}_seg.nii.gz
sct_deepseg_sc -i ${file_ses2_onlyfile}_res.nii.gz -c t1 -o ${file_ses2_onlyfile}_seg.nii.gz

# Perform registration ses-01 --> ses-02
# NOTE: We are passing the un-resampled / original ses-01 as the input to avoid double interpolation
# NOTE: We do NOT need to co-register the brain, because it was ALREADY co-registered by the MS-CHALLENGE team.
sct_register_multimodal -i ${file_ses1}.nii.gz -iseg ${file_ses1_onlyfile}_seg.nii.gz -d ${file_ses2_onlyfile}_res.nii.gz -dseg ${file_ses2_onlyfile}_seg.nii.gz -o ${file_ses1_onlyfile}_reg.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares,smooth=3

# Dilate spinal cord mask
sct_maths -i ${file_ses2_onlyfile}_seg.nii.gz -dilate 5 -shape ball -o ${file_ses2_onlyfile}_seg_dilate.nii.gz

# Brain extraction
if [[ $BRAIN_EXTRACTION_METHOD == "anima" ]]; then
  python $ANIMA_SCRIPTS_PUBLIC_PATH/brain_extraction/animaAtlasBasedBrainExtraction.py -i ${file_ses2_onlyfile}_res.nii.gz --mask brain_mask.nii.gz
elif [[ $BRAIN_EXTRACTION_METHOD == "bet2" ]]; then
  bet2 ${file_ses2_onlyfile}_res.nii.gz brain -m
elif [[ $BRAIN_EXTRACTION_METHOD == "anima+bet2" ]]; then
  python $ANIMA_SCRIPTS_PUBLIC_PATH/brain_extraction/animaAtlasBasedBrainExtraction.py -i ${file_ses2_onlyfile}_res.nii.gz --mask brain_anima_mask.nii.gz
  bet2 ${file_ses2_onlyfile}_res.nii.gz brain_bet2 -m
  # Sum two brain masks and binarize
  sct_maths -i brain_anima_mask.nii.gz -add brain_bet2_mask.nii.gz -o brain_mask.nii.gz
  sct_maths -i brain_mask.nii.gz -bin 0.5 -o brain_mask.nii.gz
else
  echo "Brain extraction method = ${BRAIN_EXTRACTION_METHOD} is not recognized!"
  exit 1
fi

# Dilate brain mask
sct_maths -i brain_mask.nii.gz -dilate 5 -shape ball -o brain_mask_dilate.nii.gz

# Sum brain mask and spinal cord mask and binarize
sct_maths -i brain_mask_dilate.nii.gz -add ${file_ses2_onlyfile}_seg_dilate.nii.gz -o brain_cord_mask.nii.gz
sct_maths -i brain_cord_mask.nii.gz -bin 0.5 -o brain_cord_mask.nii.gz

# Compute the bounding box coordinates of the brain + SC mask for cropping the VOI
# NOTE: `fslstats -w returns the smallest ROI <xmin> <xsize> <ymin> <ysize> <zmin> <zsize> <tmin> <tsize> containing nonzero voxels
brain_cord_mask_bbox_coords=$(fslstats brain_cord_mask.nii.gz -w)

# Finer registration with ANTs
# TODO: use initial transform with -r flag
antsRegistration -d 3 -m CC[${file_ses2_onlyfile}_res.nii.gz, ${file_ses1_onlyfile}_reg.nii.gz, 1, 4, Regular, 1] -t SyN[0.5] -c 20x10x2 -s 0x0x1 -f 8x4x2 -n BSpline -x brain_cord_mask.nii.gz -o [warp_, ${file_ses1_onlyfile}_reg-brain.nii.gz] -v 1

# Apply the brain + SC mask to the final forms of both sessions
fslmaths ${file_ses1_onlyfile}_reg-brain.nii.gz -mas brain_cord_mask.nii.gz ${file_ses1_onlyfile}_reg-brain_masked.nii.gz
fslmaths ${file_ses2_onlyfile}_res.nii.gz -mas brain_cord_mask.nii.gz ${file_ses2_onlyfile}_res_masked.nii.gz

# Remove bias from both sessions
$ANIMA_BINARIES_PATH/animaN4BiasCorrection -i ${file_ses1_onlyfile}_reg-brain_masked.nii.gz -o ${file_ses1_onlyfile}_reg-brain_masked.nii.gz -B 0.3
$ANIMA_BINARIES_PATH/animaN4BiasCorrection -i ${file_ses2_onlyfile}_res_masked.nii.gz -o ${file_ses2_onlyfile}_res_masked.nii.gz -B 0.3

# Crop the VOI based on the brain + SC mask to minimize the input image size
fslroi ${file_ses1_onlyfile}_reg-brain_masked.nii.gz ${file_ses1_onlyfile}_reg-brain_masked.nii.gz $brain_cord_mask_bbox_coords
fslroi ${file_ses2_onlyfile}_res_masked.nii.gz ${file_ses2_onlyfile}_res_masked.nii.gz $brain_cord_mask_bbox_coords

# The following files are the final images for session 1 and 2, which will be inputted to the model
# ses01 -> ${file_ses1_onlyfile}_reg-brain_masked.nii.gz
# ses02 -> ${file_ses2_onlyfile}_res_masked.nii.gz

# (2) Go to subject folder for segmentation GTs
cd $PATH_DATA_PROCESSED/derivatives/labels/$SUBJECT

file_gt1_onlyfile="${SUBJECT}_ses-02_FLAIR_lesion-manual-rater1"
file_gt1="ses-02/anat/${SUBJECT}_ses-02_FLAIR_lesion-manual-rater1"
file_gt2_onlyfile="${SUBJECT}_ses-02_FLAIR_lesion-manual-rater2"
file_gt2="ses-02/anat/${SUBJECT}_ses-02_FLAIR_lesion-manual-rater2"
file_gt3_onlyfile="${SUBJECT}_ses-02_FLAIR_lesion-manual-rater3"
file_gt3="ses-02/anat/${SUBJECT}_ses-02_FLAIR_lesion-manual-rater3"
file_gt4_onlyfile="${SUBJECT}_ses-02_FLAIR_lesion-manual-rater4"
file_gt4="ses-02/anat/${SUBJECT}_ses-02_FLAIR_lesion-manual-rater4"
file_gtc_onlyfile="${SUBJECT}_ses-02_FLAIR_seg-lesion"
file_gtc="ses-02/anat/${SUBJECT}_ses-02_FLAIR_seg-lesion"

# Resample segmentation GTs to isotropic 0.75mm x 0.75mm x 0.75mm voxel dimensions
flirt -in ${file_gt1}.nii.gz -ref ${file_gt1}.nii.gz -applyisoxfm 0.75 -nosearch -out ${file_gt1_onlyfile}_res.nii.gz
flirt -in ${file_gt2}.nii.gz -ref ${file_gt2}.nii.gz -applyisoxfm 0.75 -nosearch -out ${file_gt2_onlyfile}_res.nii.gz
flirt -in ${file_gt3}.nii.gz -ref ${file_gt3}.nii.gz -applyisoxfm 0.75 -nosearch -out ${file_gt3_onlyfile}_res.nii.gz
flirt -in ${file_gt4}.nii.gz -ref ${file_gt4}.nii.gz -applyisoxfm 0.75 -nosearch -out ${file_gt4_onlyfile}_res.nii.gz
flirt -in ${file_gtc}.nii.gz -ref ${file_gtc}.nii.gz -applyisoxfm 0.75 -nosearch -out ${file_gtc_onlyfile}_res.nii.gz

# Apply the brain + SC mask to the final forms of all segmentation GTs
fslmaths ${file_gt1_onlyfile}_res.nii.gz -mas $PATH_DATA_PROCESSED/$SUBJECT/brain_cord_mask.nii.gz ${file_gt1_onlyfile}_res_masked.nii.gz
fslmaths ${file_gt2_onlyfile}_res.nii.gz -mas $PATH_DATA_PROCESSED/$SUBJECT/brain_cord_mask.nii.gz ${file_gt2_onlyfile}_res_masked.nii.gz
fslmaths ${file_gt3_onlyfile}_res.nii.gz -mas $PATH_DATA_PROCESSED/$SUBJECT/brain_cord_mask.nii.gz ${file_gt3_onlyfile}_res_masked.nii.gz
fslmaths ${file_gt4_onlyfile}_res.nii.gz -mas $PATH_DATA_PROCESSED/$SUBJECT/brain_cord_mask.nii.gz ${file_gt4_onlyfile}_res_masked.nii.gz
fslmaths ${file_gtc_onlyfile}_res.nii.gz -mas $PATH_DATA_PROCESSED/$SUBJECT/brain_cord_mask.nii.gz ${file_gtc_onlyfile}_res_masked.nii.gz

# Crop the VOI based on the brain + SC mask to minimize the GT image size
fslroi ${file_gt1_onlyfile}_res_masked.nii.gz ${file_gt1_onlyfile}_res_masked.nii.gz $brain_cord_mask_bbox_coords
fslroi ${file_gt2_onlyfile}_res_masked.nii.gz ${file_gt2_onlyfile}_res_masked.nii.gz $brain_cord_mask_bbox_coords
fslroi ${file_gt3_onlyfile}_res_masked.nii.gz ${file_gt3_onlyfile}_res_masked.nii.gz $brain_cord_mask_bbox_coords
fslroi ${file_gt4_onlyfile}_res_masked.nii.gz ${file_gt4_onlyfile}_res_masked.nii.gz $brain_cord_mask_bbox_coords
fslroi ${file_gtc_onlyfile}_res_masked.nii.gz ${file_gtc_onlyfile}_res_masked.nii.gz $brain_cord_mask_bbox_coords

# The following files are the final GTs for all experts, which will be inputted to the model
# rater1 -> ${file_gt1_onlyfile}_res_masked.nii.gz
# rater2 -> ${file_gt2_onlyfile}_res_masked.nii.gz
# rater3 -> ${file_gt3_onlyfile}_res_masked.nii.gz
# rater4 -> ${file_gt4_onlyfile}_res_masked.nii.gz
# consensus -> ${file_gtc_onlyfile}_res_masked.nii.gz

# Go back to parent folder (i.e. get ready for next subject call!)
cd $PATH_DATA_PROCESSED

# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
FILES_TO_CHECK=(
"brain_cord_mask.nii.gz"
"brain_mask_dilate.nii.gz"
"brain_mask.nii.gz"
"brain.nii.gz"
"${file_ses1_onlyfile}_reg-brain_masked.nii.gz"
"${file_ses1_onlyfile}_reg-brain.nii.gz"
"${file_ses1_onlyfile}_reg_inv.nii.gz"
"${file_ses1_onlyfile}_reg.nii.gz"
"${file_ses1_onlyfile}_res.nii.gz"
"${file_ses1_onlyfile}_seg.nii.gz"
"${file_ses2_onlyfile}_res_masked.nii.gz"
"${file_ses2_onlyfile}_res.nii.gz"
"${file_ses2_onlyfile}_seg_dilate.nii.gz"
"${file_ses2_onlyfile}_seg.nii.gz"
"warp_0InverseWarp.nii.gz"
"warp_0Warp.nii.gz"
"warp_${file_ses1_onlyfile}2${file_ses2_onlyfile}_res.nii.gz"
"warp_${file_ses2_onlyfile}_res2${file_ses1_onlyfile}.nii.gz"
)

for file in "${FILES_TO_CHECK[@]}"; do
  if [[ ! -e ${SUBJECT}/${file} ]]; then
    echo "${SUBJECT}/${file} does not exist" >> $PATH_LOG/_error_check_output_files.log
  fi
done

# Go back to the root output path
cd $PATH_OUTPUT

# Create and populate clean data processed folder for training
PATH_DATA_PROCESSED_CLEAN="${PATH_DATA_PROCESSED}_clean"


# Copy over required BIDs files
mkdir -p $PATH_DATA_PROCESSED_CLEAN $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/ses-01/ $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/ses-01/anat
mkdir -p $PATH_DATA_PROCESSED_CLEAN $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/ses-02/ $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/ses-02/anat

rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/participants.* $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/README $PATH_DATA_PROCESSED_CLEAN/
rsync -avzh $PATH_DATA_PROCESSED/dataset_description.json $PATH_DATA_PROCESSED_CLEAN/derivatives/

# Refer to lines 142-144 to know which preprocessed files to copy 
# MR images for ses-01
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/${file_ses1_onlyfile}_reg-brain_masked.nii.gz $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/ses-01/anat/${file_ses1_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/ses-01/anat/${file_ses1_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/ses-01/anat/${file_ses1_onlyfile}.json
# MR images for ses-02
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/${file_ses2_onlyfile}_res_masked.nii.gz $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/ses-02/anat/${file_ses2_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/${SUBJECT}/ses-02/anat/${file_ses2_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/${SUBJECT}/ses-02/anat/${file_ses2_onlyfile}.json

# Refer to lines 181-186 to know which preprocessed GTs to copy 
# Note that the original data has the GTs from the 2nd timepoint ie ses-02
mkdir -p $PATH_DATA_PROCESSED_CLEAN/derivatives $PATH_DATA_PROCESSED_CLEAN/derivatives/labels $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT} $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-02 $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-02/anat

# Preprocessed GT from rater 1
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/${file_gt1_onlyfile}_res_masked.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gt1_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gt1_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gt1_onlyfile}.json
# Preprocessed GT from rater 2
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/${file_gt2_onlyfile}_res_masked.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gt2_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gt2_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gt2_onlyfile}.json
# Preprocessed GT from rater 3
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/${file_gt3_onlyfile}_res_masked.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gt3_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gt3_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gt3_onlyfile}.json
# Preprocessed GT from rater 4
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/${file_gt4_onlyfile}_res_masked.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gt4_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gt4_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gt4_onlyfile}.json
# Preprocessed consensus GT 
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/${file_gtc_onlyfile}_res_masked.nii.gz $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gtc_onlyfile}.nii.gz
rsync -avzh $PATH_DATA_PROCESSED/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gtc_onlyfile}.json $PATH_DATA_PROCESSED_CLEAN/derivatives/labels/${SUBJECT}/ses-02/anat/${file_gtc_onlyfile}.json


# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"
