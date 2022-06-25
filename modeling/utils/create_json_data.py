import os
import json
import pandas as pd
import argparse
from tqdm import tqdm
import random
import numpy as np
from fold_generator import FoldGenerator

# Code for creating "k" json files containing the respective dataset splits for k-fold cross validation 
# Usage:
# python create_json_data.py -se 30 -ncv 5 -dr <root path of the dataset>
# Creates "k" files named `dataset_fold-{X}.json` inside the root dataset folder

root = "/home/GRAMES.POLYMTL.CA/u114716/duke/temp/muena/ms-challenge-2021_preprocessed/data_processed_clean"

parser = argparse.ArgumentParser(description='Code for creating k-fold splits of the ms-challenge-2021 dataset.')

parser.add_argument('-se', '--seed', default=42, type=int, help="Seed for reproducibility")
parser.add_argument('-ncv', '--num_cv_folds', default=5, type=int, help="To create a k-fold dataset for cross validation")
parser.add_argument('-dr', '--data_root', default=root, type=str, help='Path to the data set directory')

args = parser.parse_args()

# Get all subjects
subjects_df = pd.read_csv(os.path.join(args.data_root, 'participants.tsv'), sep='\t')
subjects = subjects_df['participant_id'].values.tolist()
# print(subjects)

seed = args.seed
num_cv_folds = args.num_cv_folds    # for 100 subjects, performs a 60-20-20 split with num_cv_plots

# returns a nested list of length (num_cv_folds), each element (again, a list) consisting of 
# train, val, test indices and the fold number
names_list = FoldGenerator(seed, num_cv_folds, len_data=len(subjects)).get_fold_names()

for fold in range(num_cv_folds):

    train_ix, val_ix, test_ix, fold_num = names_list[fold]
    training_subjects = [subjects[tr_ix] for tr_ix in train_ix]
    validation_subjects = [subjects[v_ix] for v_ix in val_ix]
    test_subjects = [subjects[te_ix] for te_ix in test_ix]
    # print(training_subjects, "\n",validation_subjects, "\n",test_subjects)

    # keys to be defined in the dataset_0.json
    params = {}
    params["description"] = "MICCAI MS New Lesion Segmentation Challenge 2021"
    params["labels"] = {
        "0": "background",
        "1": "ms-lesion"
        }
    params["license"] = "nk"
    params["modality"] = {
        "0": "MRI"
        }
    params["name"] = f"ms-challenge-2021 data fold-{fold_num}"
    params["numTest"] = len(test_subjects)
    params["numTraining"] = len(training_subjects) + len(validation_subjects)
    params["reference"] = "MICCAI"
    params["tensorImageSize"] = "3D"


    train_val_subjects_dict = {
        "training": training_subjects,
        "validation": validation_subjects,
    } 
    test_subjects_dict =  {"test": test_subjects}


    # run loop for training and validation subjects
    for name, subs_list in train_val_subjects_dict.items():

        temp_list = []
        for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
            
            temp_data = {}
            # Read-in input volumes
            # e.g. file:  sub-005_ses-02_FLAIR.nii.gz
            ses01_flair = os.path.join(args.data_root, subject, 'ses-01', 'anat', '%s_ses-01_FLAIR.nii.gz' % subject)
            ses02_flair = os.path.join(args.data_root, subject, 'ses-02', 'anat', '%s_ses-02_FLAIR.nii.gz' % subject)

            # Read-in GT volumes (using the consensus GT for now)
            gtc = os.path.join(args.data_root, 'derivatives', 'labels', subject, 'ses-02', 'anat', '%s_ses-02_FLAIR_seg-lesion.nii.gz' % subject)
            # GT volumes from other raters
            # e.g file: sub-005_ses-02_FLAIR_lesion-manual-rater1.nii.gz
            gt_rater1 = os.path.join(args.data_root, 'derivatives', 'labels', subject, 'ses-02', 'anat', '%s_ses-02_FLAIR_lesion-manual-rater1.nii.gz' % subject)
            gt_rater2 = os.path.join(args.data_root, 'derivatives', 'labels', subject, 'ses-02', 'anat', '%s_ses-02_FLAIR_lesion-manual-rater2.nii.gz' % subject)
            gt_rater3 = os.path.join(args.data_root, 'derivatives', 'labels', subject, 'ses-02', 'anat', '%s_ses-02_FLAIR_lesion-manual-rater3.nii.gz' % subject)
            gt_rater4 = os.path.join(args.data_root, 'derivatives', 'labels', subject, 'ses-02', 'anat', '%s_ses-02_FLAIR_lesion-manual-rater4.nii.gz' % subject)
                
            # store in a temp dictionary
            temp_data["image_ses1"] = ses01_flair.replace(args.data_root+"/", '') # .strip(root)
            temp_data["image_ses2"] = ses02_flair.replace(args.data_root+"/", '') # .strip(root)
            
            temp_data["label_c"] = gtc.replace(args.data_root+"/", '')       # .strip(root)
            temp_data["label_1"] = gt_rater1.replace(args.data_root+"/", '') # .strip(root)
            temp_data["label_2"] = gt_rater2.replace(args.data_root+"/", '') # .strip(root)
            temp_data["label_3"] = gt_rater3.replace(args.data_root+"/", '') # .strip(root)
            temp_data["label_4"] = gt_rater4.replace(args.data_root+"/", '') # .strip(root)

            temp_list.append(temp_data)
        
        params[name] = temp_list


    # run separte loop for testing "without" the labels
    for name, subs_list in test_subjects_dict.items():
        temp_list = []
        for subject_no, subject in enumerate(tqdm(subs_list, desc='Loading Volumes')):
        
            temp_data = {}
            # Read-in input volumes
            # e.g. file:  sub-005_ses-02_FLAIR.nii.gz
            ses01_flair = os.path.join(args.data_root, subject, 'ses-01', 'anat', '%s_ses-01_FLAIR.nii.gz' % subject)
            ses02_flair = os.path.join(args.data_root, subject, 'ses-02', 'anat', '%s_ses-02_FLAIR.nii.gz' % subject)

            # store in a temp dictionary
            temp_data["image_ses1"] = ses01_flair.replace(args.data_root+"/", '') # .strip(root)
            temp_data["image_ses2"] = ses02_flair.replace(args.data_root+"/", '') # .strip(root)
                
            temp_list.append(temp_data)
        
        params[name] = temp_list

    final_json = json.dumps(params, indent=4, sort_keys=True)
    jsonFile = open(args.data_root + "/" + f"dataset_fold-{fold_num}.json", "w")
    jsonFile.write(final_json)
    jsonFile.close()




    


