import os
import shutil
import json
import argparse

def get_parameters():
    parser = argparse.ArgumentParser(description='This script is curating dataset msseg_challenge_2021 to BIDS')
    parser.add_argument("-d", "--data",
                        help="Path to folder containing the dataset to be curated",
                        required=True)
    args = parser.parse_args()
    return args

def main(root_data):
    output_data = os.path.join(root_data + '_curated')

    #Remove macosx ds_store
    os.system("find " + root_data + " -name '.DS_Store' -type f -delete")

    if os.path.isdir(output_data):
        shutil.rmtree(output_data)
    os.makedirs(output_data)

    # Path for raw images:
    # /sub-XXX/anat/sub-XXX_ses-0X_acq-middlespace_FLAIR.nii.gz
    dict_images = { "flair_time01_on_middle_space.nii.gz":"_ses-01_acq-middlespace_FLAIR.nii.gz",
                    "flair_time02_on_middle_space.nii.gz":"_ses-02_acq-middlespace_FLAIR.nii.gz",
                  }
    # Path for derivative images:
    # /derivatives/labels/sub-XXX/anat/sub-XXX_acq-expertX_lesion-manual.nii.gz
    # /derivatives/labels/sub-XXX/anat/sub-XXX_seg-lesion.nii.gz
    dict_der =  {   "ground_truth_expert1.nii.gz":"_acq-expert1_lesion-manual.nii.gz",
                    "ground_truth_expert2.nii.gz":"_acq-expert2_lesion-manual.nii.gz",
                    "ground_truth_expert3.nii.gz":"_acq-expert3_lesion-manual.nii.gz",
                    "ground_truth_expert4.nii.gz":"_acq-expert4_lesion-manual.nii.gz",
                    "ground_truth.nii.gz":"_seg-lesion.nii.gz"
                 }

    for sub in os.listdir(root_data):
        subid_bids = "sub-" + sub
        path_sub = os.path.join(root_data,sub)
        list_files_sub = os.listdir(path_sub)
        for file in list_files_sub:
            path_file_in = os.path.join(path_sub,file)
            flag_der = False
            if file.startswith('ground_truth'):
                path_subid_bids_dir_out = os.path.join(output_data, 'derivatives', 'labels', subid_bids, 'anat')
                flag_der = True
            else:
                path_subid_bids_dir_out = os.path.join(output_data,subid_bids,'anat')
            if not os.path.isdir(path_subid_bids_dir_out):
                os.makedirs(path_subid_bids_dir_out)
            if flag_der == False:
                path_file_out = os.path.join(path_subid_bids_dir_out, subid_bids + dict_images[file])
            else:
                path_file_out = os.path.join(path_subid_bids_dir_out, subid_bids + dict_der[file])
            shutil.copy(path_file_in, path_file_out)

        for dirName, subdirList, fileList in os.walk(output_data):
            for file in fileList:
                if file.endswith('.nii.gz') :
                    originalFilePath = os.path.join(dirName,file)
                    jsonSidecarPath = os.path.join(dirName,file.split(".")[0]+'.json')
                    if os.path.exists(jsonSidecarPath) == False:
                        print ("Missing: " + jsonSidecarPath)
                        os.system('touch ' + jsonSidecarPath)

    sub_list = os.listdir(output_data)
    sub_list.remove ('derivatives')

    sub_list.sort()

    import csv

    participants = []
    for subject in sub_list:
        row_sub = []
        row_sub.append(subject)
        row_sub.append('n/a')
        row_sub.append('n/a')
        participants.append(row_sub)

    print (participants)
    with open(output_data+'/participants.tsv', 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(["participant_id", "sex", "age"])
        for item in participants:
            tsv_writer.writerow(item)

if __name__ == "__main__":
    args = get_parameters()
    main(args.data)
