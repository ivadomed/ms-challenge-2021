#!/bin/bash
#
# Preprocess data.
#
# Dependencies:
# - bet2 (FSL) <TODO: VERSION>
# - SCT <TODO: VERSION>
# - ANTs <TODO: VERSION>
# - animaBrainExtraction (anima) <TODO: VERSION>
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
ANIMA_SCRIPTS_PUBLIC_PATH=$(grep "anima-scripts-public-root" ~/.anima/config.txt | sed 's/.* = //')

# get starting time:
start=`date +%s`


# SCRIPT STARTS HERE
# ==============================================================================
# Display useful info for the log, such as SCT version, RAM and CPU cores available
sct_check_dependencies -short

# Go to folder where data will be copied and processed
cd $PATH_DATA_PROCESSED
# Copy list of participants in processed data folder
if [[ ! -f "participants.tsv" ]]; then
  rsync -avzh $PATH_DATA/participants.tsv .
fi
# Copy source images
rsync -avzh $PATH_DATA/$SUBJECT .
# Go to subject folder
cd ${SUBJECT}

# TODO: re-think how file variable is defined-- not clean to have folders in there
file_ses1_onlyfile="${SUBJECT}_ses-01_FLAIR"
file_ses1="ses-01/anat/${SUBJECT}_ses-01_FLAIR"
file_ses2_onlyfile="${SUBJECT}_ses-02_FLAIR"
file_ses2="ses-02/anat/${SUBJECT}_ses-02_FLAIR"

# Spinal cord extraction
sct_deepseg_sc -i ${file_ses1}.nii.gz -c t1
sct_deepseg_sc -i ${file_ses2}.nii.gz -c t1

# Perform registration ses-01 --> ses-02
sct_register_multimodal -i ${file_ses1}.nii.gz -iseg ${file_ses1_onlyfile}_seg.nii.gz -d ${file_ses2}.nii.gz -dseg ${file_ses2_onlyfile}_seg.nii.gz -param step=1,type=seg,algo=slicereg,metric=MeanSquares,smooth=3

# Dilate spinal cord mask
sct_maths -i ${file_ses2_onlyfile}_seg.nii.gz -dilate 5 -shape ball -o ${file_ses2_onlyfile}_seg_dilate.nii.gz

# Brain extraction
if [[ $BRAIN_EXTRACTION_METHOD == "anima" ]]; then
  python $ANIMA_SCRIPTS_PUBLIC_PATH/brain_extraction/animaAtlasBasedBrainExtraction.py -i ${file_ses2}.nii.gz --mask brain_mask.nii.gz
elif [[ $BRAIN_EXTRACTION_METHOD == "bet2" ]]; then
  bet2 ${file_ses2}.nii.gz brain -m
elif [[ $BRAIN_EXTRACTION_METHOD == "anima+bet2" ]]; then
  python $ANIMA_SCRIPTS_PUBLIC_PATH/brain_extraction/animaAtlasBasedBrainExtraction.py -i ${file_ses2}.nii.gz --mask brain_anima_mask.nii.gz
  bet2 ${file_ses2}.nii.gz brain_bet2 -m
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

# Finer registration with ANTs
# TODO: use initial transform with -r flag
antsRegistration -d 3 -m CC[${file_ses2}.nii.gz, ${file_ses1_onlyfile}_reg.nii.gz, 1, 4, Regular, 1] -t SyN[0.5] -c 20x10x2 -s 0x0x1 -f 8x4x2 -n BSpline -x brain_cord_mask.nii.gz -o [warp_, ${file_ses1_onlyfile}_reg-brain.nii.gz] -v 1

# Go back to parent folder
cd ..

# Verify presence of output files and write log file if error
# ------------------------------------------------------------------------------
FILES_TO_CHECK=(
"brain_cord_mask.nii.gz"
"brain_mask_dilate.nii.gz"
"brain_mask.nii.gz"
"brain.nii.gz"
"${file_ses1_onlyfile}_reg-brain.nii.gz"
"${file_ses1_onlyfile}_reg.nii.gz"
"${file_ses1_onlyfile}_seg.nii.gz"
"${file_ses2_onlyfile}_reg.nii.gz"
"${file_ses2_onlyfile}_seg_dilate.nii.gz"
"${file_ses2_onlyfile}_seg.nii.gz"
"warp_0InverseWarp.nii.gz"
"warp_0Warp.nii.gz"
"warp_${file_ses1_onlyfile}2${file_ses2_onlyfile}.nii.gz"
"warp_${file_ses2_onlyfile}2${file_ses1_onlyfile}.nii.gz"
)

for file in "${FILES_TO_CHECK[@]}"; do
  if [[ ! -e ${SUBJECT}/${file} ]]; then
    echo "${SUBJECT}/${file} does not exist" >> $PATH_LOG/_error_check_output_files.log
  fi
done

# Display useful info for the log
end=`date +%s`
runtime=$((end-start))
echo
echo "~~~"
echo "SCT version: `sct_version`"
echo "Ran on:      `uname -nsr`"
echo "Duration:    $(($runtime / 3600))hrs $((($runtime / 60) % 60))min $(($runtime % 60))sec"
echo "~~~"

