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

    # Remove macosx ds_store
    os.system("find " + root_data + " -name '.DS_Store' -type f -delete")

    if os.path.isdir(output_data):
        shutil.rmtree(output_data)
    os.makedirs(output_data)

    # Path for raw images:
    # /sub-XXX/anat/sub-XXX_ses-0X_acq-middlespace_FLAIR.nii.gz
    image_end_name = "acq_FLAIR.nii.gz"
    dict_images = {"flair_time01_on_middle_space.nii.gz": "ses-01",
                   "flair_time02_on_middle_space.nii.gz": "ses-02",
                   }

    # Path for derivative images:
    # /derivatives/labels/sub-XXX/anat/sub-XXX_acq-middlespace_lesion-manual-rater1.nii.gz
    # /derivatives/labels/sub-XXX/anat/sub-XXX_acq-middlespace_seg-lesion.nii.gz
    dict_der = {"ground_truth_expert1.nii.gz": "_lesion-manual-rater1.nii.gz",
                "ground_truth_expert2.nii.gz": "_lesion-manual-rater2.nii.gz",
                "ground_truth_expert3.nii.gz": "_lesion-manual-rater3.nii.gz",
                "ground_truth_expert4.nii.gz": "_lesion-manual-rater4.nii.gz",
                "ground_truth.nii.gz": "_seg-lesion.nii.gz"
                }

    subjects = os.listdir(root_data)
    #subjects = subjects.remove("_curated") # For some reason that is beyond me this fails
    subjects = [x for x in subjects if x != '_curated']
    for sub in subjects:
        subid_bids = "sub-" + sub
        path_sub = os.path.join(root_data, sub)
        list_files_sub = os.listdir(path_sub)
        for file in list_files_sub:
            if '_sc_mask' not in file:
                path_file_in = os.path.join(path_sub, file)
                flag_der = False
                if file.startswith('ground_truth'):
                    path_subid_bids_dir_out = os.path.join(output_data, 'derivatives', 'labels', subid_bids, "ses-02", 'anat')
                    flag_der = True
                else:
                    path_subid_bids_dir_out = os.path.join(output_data, subid_bids, dict_images[file], 'anat')
                if not os.path.isdir(path_subid_bids_dir_out):
                    os.makedirs(path_subid_bids_dir_out)
                if flag_der is False:
                    path_file_out = os.path.join(path_subid_bids_dir_out,
                                                 '{0}_{1}_{2}'.format(subid_bids, dict_images[file], image_end_name))
                else:
                    path_file_out = os.path.join(path_subid_bids_dir_out, subid_bids + "_ses-02_acq_FLAIR" + dict_der[file])
                shutil.copy(path_file_in, path_file_out)

        for dirName, subdirList, fileList in os.walk(output_data):
            for file in fileList:
                if file.endswith('.nii.gz'):
                    originalfilepath = os.path.join(dirName, file)
                    jsonsidecarpath = os.path.join(dirName, file.split(".")[0] + '.json')
                    if os.path.exists(jsonsidecarpath) is False:
                        print("Missing: " + jsonsidecarpath)
                        os.system('touch ' + jsonsidecarpath)

    sub_list = os.listdir(output_data)
    sub_list.remove('derivatives')

    sub_list.sort()

    # Write additional files
    import csv
    participants = []
    for subject in sub_list:
        row_sub = []
        row_sub.append(subject)
        row_sub.append('n/a')
        row_sub.append('n/a')
        participants.append(row_sub)

    print(participants)
    with open(output_data + '/participants.tsv', 'w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(["participant_id", "sex", "age"])
        for item in participants:
            tsv_writer.writerow(item)

    with open(output_data + '/README', 'w') as readme_file:
        readme_file.write('Dataset for msseg_challenge_2021.')

    data_json = {"participant_id": {
                "Description": "Unique ID",
                "LongName": "Participant ID"
                },
                "sex": {
                    "Description": "M or F",
                    "LongName": "Participant gender"
                },
                "age": {
                    "Description": "yy",
                    "LongName": "Participant age"
                }
                }
    with open(output_data + '/participants.json', 'w') as json_file:
        json.dump(data_json, json_file, indent=4)

    # Dataset description in the output folder
    dataset_description = {"BIDSVersion": "1.6.0",
                           "Name": "ms_challenge_2021"
                           }

    with open(output_data + '/dataset_description.json', 'w') as json_file:
        json.dump(dataset_description, json_file, indent=4)


    # Dataset description in the derivatives folder
    dataset_description = {"Name": "Example dataset",
                            "BIDSVersion": "1.6.0",
                            "PipelineDescription": {"Name": "Example pipeline"}
                           }

    with open(output_data + '/derivatives/dataset_description.json', 'w') as json_file:
        json.dump(dataset_description, json_file, indent=4)


if __name__ == "__main__":
    args = get_parameters()
    main(args.data)
