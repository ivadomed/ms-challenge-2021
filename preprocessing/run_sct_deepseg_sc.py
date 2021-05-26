import os
import argparse

# Argument parsing
parser = argparse.ArgumentParser(
    description="""Run sct_deepseg_sc.""", formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument('-s', '--source', type=str, required=True, help="""Input folder.
The folder must follow this structure:
/input/folder/
 013
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

flairs = ['flair_time01_on_middle_space.nii.gz', 'flair_time02_on_middle_space.nii.gz']

for patientName in os.listdir(src_folder):

    src_patient_folder = os.path.join(src_folder, patientName)

    if not os.path.isdir(src_patient_folder): continue

    print("Running SC extraction of patient " + patientName + "...")

    for flairName in flairs:
        src_flair = os.path.join(src_patient_folder, flairName)
        # Output fname
        output_fname = os.path.join(src_patient_folder, flairName.split('.nii.gz')[0]+'_sc_mask.nii.gz')
        # Run command
        os.system("sct_deepseg_sc -i {} -c t1 -o {}".format(src_flair, output_fname))
        print('\tSaving difference in {}'.format(output_fname))
