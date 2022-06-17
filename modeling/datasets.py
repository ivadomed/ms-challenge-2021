"""Implements datasets for MSSeg2021 Challenge."""
import os
import subprocess
from tqdm import tqdm
import random
from collections import defaultdict
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import nibabel as nib


import torch
from torch.utils.data import Dataset

from ivadomed.transforms import Resample, CenterCrop, RandomAffine, ElasticTransform, NormalizeInstance
from torchio.transforms import Compose, RandomBiasField


# ---------------------------- Helpers Implementation -----------------------------
def volume2subvolumes(volume, subvolume_size, stride_size):
    """Converts 3D volumes into 3D subvolumes; works with PyTorch tensors."""
    subvolumes = []
    assert volume.ndim == 3

    for x in range(0, (volume.shape[0] - subvolume_size[0]) + 1, stride_size[0]):
        for y in range(0, (volume.shape[1] - subvolume_size[1]) + 1, stride_size[1]):
            for z in range(0, (volume.shape[2] - subvolume_size[2]) + 1, stride_size[2]):
                subvolume = volume[x: x + subvolume_size[0],
                                   y: y + subvolume_size[1],
                                   z: z + subvolume_size[2]]
                subvolumes.append(subvolume)

    return subvolumes


def subvolumes2volume(subvolumes, volume_size):
    """Converts list of 3D subvolumes into 3D volumes; works with NumPy arrays."""
    assert len(volume_size) == 3
    subvolume_size = subvolumes[0].shape
    volume = np.zeros(volume_size)

    num_subvolumes_per_dim = [volume_size[i] // subvolume_size[i] for i in range(3)]

    for i, x in enumerate(range(0, (volume_size[0] - subvolume_size[0]) + 1, subvolume_size[0])):
        for j, y in enumerate(range(0, (volume_size[1] - subvolume_size[1]) + 1, subvolume_size[1])):
            for k, z in enumerate(range(0, (volume_size[2] - subvolume_size[2]) + 1, subvolume_size[2])):
                # Get the subvolume index for the corresponding spatial location
                subvolume_index = i * np.prod(num_subvolumes_per_dim[1:]) + j * num_subvolumes_per_dim[2] + k
                # Fill in the volume with the corresponding subvolume
                volume[x: x + subvolume_size[0],
                       y: y + subvolume_size[1],
                       z: z + subvolume_size[2]] = subvolumes[subvolume_index]

    return volume


def subvolume2patches(subvolume, patch_size):
    """Extracts 3D patches from 3D subvolumes; works with PyTorch tensors."""
    patches = []
    assert subvolume.ndim == 3

    for x in range(0, (subvolume.shape[0] - patch_size[0]) + 1, patch_size[0]):
        for y in range(0, (subvolume.shape[1] - patch_size[1]) + 1, patch_size[1]):
            for z in range(0, (subvolume.shape[2] - patch_size[2]) + 1, patch_size[2]):
                patch = subvolume[x: x + patch_size[0],
                                  y: y + patch_size[1],
                                  z: z + patch_size[2]]

                patches.append(patch)

    num_patches = len(patches)
    patches = np.array(patches)
    assert patches.shape == (num_patches, *patch_size)

    return patches


def patches2subvolume(patches, subvolume_size):
    """Converts list of 3D patches into 3D subvolumes; works with NumPy arrays."""
    assert len(subvolume_size) == 3
    patch_size = patches[0].shape
    subvolume = np.zeros(subvolume_size)

    num_patches_per_dim = [subvolume_size[i] // patch_size[i] for i in range(3)]

    for i, x in enumerate(range(0, (subvolume_size[0] - patch_size[0]) + 1, patch_size[0])):
        for j, y in enumerate(range(0, (subvolume_size[1] - patch_size[1]) + 1, patch_size[1])):
            for k, z in enumerate(range(0, (subvolume_size[2] - patch_size[2]) + 1, patch_size[2])):
                # Get the patch index for the corresponding spatial location
                patch_index = i * np.prod(num_patches_per_dim[1:]) + j * num_patches_per_dim[2] + k
                # Fill in the subvolume with the corresponding patch
                subvolume[x: x + patch_size[0],
                          y: y + patch_size[1],
                          z: z + patch_size[2]] = patches[patch_index]

    return subvolume


# ---------------------------- Datasets Implementation -----------------------------
class MSSeg2Dataset(Dataset):
    """
    Custom PyTorch dataset for the MSSeg2 Challenge 2021. Works only with 3D subvolumes. Implements
    training, validation, and test phases. Training and validation is utilized via the canonic
    __get_item__() function and by carefully setting the `train` parameter before
    accessing an item (e.g. via iterating over the dataloader in modeling/train.py). Test phase
    is implemented with the `test` method.

    :param (float) fraction_data: Fraction of subjects to use for the entire dataset. Helps with debugging.
    :param (float) fraction_hold_out: Fraction of subjects to hold-out for the test phase. We want
           to hold-out entire patients, as opposed to subvolumes, to have a more representative
           test with the ANIMA script `animaSegPerfAnalyzer`.
    :param (tuple) center_crop_size: The 3D center-crop size for the volumes. For now, we can
           leave this at it's default value (320, 384, 512).
    :param (tuple) subvolume_size: The 3D subvolume size to be used in training & validation.
    :param (tuple) stride_size: The 3D stride size to be used in training & validation.
    :param (tuple) patch_size: The 3D patch size to be used in training & validation. (TransUNet3D-specific)
    :param (str) gt_type: The GT to use as the target masks. Leaving it at it's default 'consensus'
           seems like the best option for now.
    :param (bool) use_patches: Set to True for TransUNet3D and to False for other models.
    :param (str) results_dir: The directory to where the save the results from test phase.
    :param (bool) visualize_test_preds: Set to True to save predictions as NIfTI files during test phase.
    :param (int) seed: Seed for reproducibility (e.g. we want the same train-val split with same seed)
    """
    def __init__(self, root, fraction_data=1.0, fraction_hold_out=0.2,
                 center_crop_size=(320, 384, 512), subvolume_size=(128, 128, 128),
                 stride_size=(64, 64, 64), patch_size=(32, 32, 32), gt_type='consensus',
                 use_patches=True, results_dir='results', visualize_test_preds=False, seed=42):
        super(MSSeg2Dataset).__init__()

        # Set / create the results path for the test phase
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        else:
            print('WARNING: results_dir=%s already exists! Files might be overwritten...' % results_dir)
        self.results_dir = results_dir
        self.visualize_test_preds = visualize_test_preds

        # Get the ANIMA binaries path
        cmd = r'''grep "^anima = " ~/.anima/config.txt | sed "s/.* = //"'''
        self.anima_binaries_path = subprocess.check_output(cmd, shell=True).decode('utf-8').strip('\n')
        print('ANIMA Binaries Path: ', self.anima_binaries_path)

        # Quick argument checks
        if not os.path.exists(root):
            raise ValueError('Specified path=%s for the challenge data can NOT be found!' % root)

        if len(center_crop_size) != 3:
            raise ValueError('The `MSChallenge3D()` expects a 3D center crop size (e.g. 512x512x512)!')
        if len(subvolume_size) != 3:
            raise ValueError('The `MSChallenge3D()` expects a 3D subvolume size (e.g. 128x128x128)!')
        if len(stride_size) != 3:
            raise ValueError('The `MSChallenge3D()` expects a 3D stride size (e.g. 64x64x64)!')
        if len(patch_size) != 3:
            raise ValueError('The `MSChallenge3D()` expects a 3D patch size (e.g. 32x32x32)!')

        if any([center_crop_size[i] < subvolume_size[i] for i in range(3)]):
            raise ValueError('The center crop size must be >= subvolume size in all dimensions!')
        if any([(center_crop_size[i] - subvolume_size[i]) % stride_size[i] != 0 for i in range(3)]):
            raise ValueError('center_crop_size - subvolume_size % stride size is NOT 0 for all dimensions!')
        if any([subvolume_size[i] < patch_size[i] for i in range(3)]):
            raise ValueError('The subvolume size must be >= patch size in all dimensions!')

        if not 0.0 < fraction_data <= 1.0:
            raise ValueError('`fraction_data` needs to be between 0.0 and 1.0!')
        if not 0.0 < fraction_hold_out <= 1.0:
            raise ValueError('`fraction_hold_out` needs to be between 0.0 and 1.0!')

        if gt_type not in ['consensus', 'expert1', 'expert2', 'expert3', 'expert4', 'random']:
            raise ValueError('gt_type=%s not recognized!' % gt_type)

        self.root = root
        self.center_crop_size = center_crop_size
        self.subvolume_size = subvolume_size
        self.stride_size = stride_size
        self.patch_size = patch_size
        self.use_patches = use_patches
        self.gt_type = gt_type
        self.train = False

        # Get all subjects
        subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
        subjects = subjects_df['participant_id'].values.tolist()

        # Only use subset of the dataset if applicable (used for debugging)
        if fraction_data != 1.0:
            subjects = subjects[:int(len(subjects) * fraction_data)]

        # Hold-out a fraction of subjects for test phase
        random.seed(seed)
        random.shuffle(subjects)
        self.subjects_hold_out = subjects[:int(len(subjects) * fraction_hold_out)]
        print('Hold-out Subjects: ', self.subjects_hold_out)

        # The rest of the subjects will be used for the train and validation phases
        subjects = subjects[int(len(subjects) * fraction_hold_out):]

        # Iterate over kept subjects (i.e. after hold-out) and extract subvolumes
        self.subvolumes, self.positive_indices = [], []
        num_negatives, num_positives = 0, 0

        for subject_no, subject in enumerate(tqdm(subjects, desc='Loading Volumes -> Preparing Subvolumes')):
            # Read-in input volumes
            ses01 = nib.load(os.path.join(root, subject, 'anat', '%s_FLAIR.nii.gz' % subject), mmap=False)
            ses02 = nib.load(os.path.join(root, subject, 'anat', '%s_T1w.nii.gz' % subject), mmap=False)

            # Read-in GT volumes (using the consensus GT for now)
            gtc = nib.load(os.path.join(root, 'derivatives', 'labels', subject, 'anat', '%s_FLAIR_seg-lesion.nii.gz' % subject), mmap=False)

            # Check if image sizes and resolutions match
            assert ses01.shape == ses02.shape == gtc.shape
            assert ses01.header['pixdim'].tolist() == ses02.header['pixdim'].tolist() == gtc.header['pixdim'].tolist()
            assert ses01.header['pixdim'].tolist()[1:4] == [0.5, 0.5, 0.5]
            # NOTE: We know that the above voxel dimensions will hold thanks to preprocessing step

            # Convert to NumPy
            ses01, ses02, gtc = ses01.get_fdata(), ses02.get_fdata(), gtc.get_fdata()

            # Apply center-cropping
            center_crop = CenterCrop(size=center_crop_size)
            ses01 = center_crop(sample=ses01, metadata={'crop_params': {}})[0]
            ses02 = center_crop(sample=ses02, metadata={'crop_params': {}})[0]
            gtc = center_crop(sample=gtc, metadata={'crop_params': {}})[0]

            # Get subvolumes from volumes and update the list
            ses01_subvolumes = volume2subvolumes(volume=ses01, subvolume_size=self.subvolume_size, stride_size=self.stride_size)
            ses02_subvolumes = volume2subvolumes(volume=ses02, subvolume_size=self.subvolume_size, stride_size=self.stride_size)
            gtc_subvolumes = volume2subvolumes(volume=gtc, subvolume_size=self.subvolume_size, stride_size=self.stride_size)

            assert len(ses01_subvolumes) == len(ses02_subvolumes) == len(gtc_subvolumes)

            for i in range(len(ses01_subvolumes)):
                subvolumes_ = {
                    'ses01': ses01_subvolumes[i],
                    'ses02': ses02_subvolumes[i],
                    'gtc': gtc_subvolumes[i],
                }

                self.subvolumes.append(subvolumes_)

                # Measure positiveness based on the consensus GT
                if np.any(gtc_subvolumes[i]):
                    self.positive_indices.append(int(subject_no * len(ses01_subvolumes) + i))
                    num_positives += 1
                else:
                    num_negatives += 1

        self.inbalance_factor = num_negatives // num_positives
        print('Factor of overall inbalance is %d!' % self.inbalance_factor)

        print('Extracted a total of %d subvolumes!' % len(self.subvolumes))

    def __getitem__(self, index):
        """Returns ses01, ses02, and GT subvolumes for the train and validation phases"""
        # Retrieve subvolumes belonging to this index for train and validation phases
        subvolumes = self.subvolumes[index]
        ses01_subvolume, ses02_subvolume, gt_subvolume = subvolumes['ses01'], subvolumes['ses02'], subvolumes['gtc']

        # Training augmentations
        if self.train:
            # (1) Apply random LR (lateral / left-right) flipping (i.e. axis=0) with P = 0.5
            if random.random() < 0.5:
                ses01_subvolume = np.flip(ses01_subvolume, axis=0)
                ses02_subvolume = np.flip(ses02_subvolume, axis=0)
                gt_subvolume = np.flip(gt_subvolume, axis=0)

            # (2.a) Apply random affine: rotation, translation, and scaling with P = 0.6
            if random.random() < 0.6:
                # NOTE: `metadata` ensures that the same affine is applied to all three subvolumes
                # NOTE: We don't want to mess the scale up too much as we use 0.5 x 0.5 x 0.5 mm^3
                random_affine = RandomAffine(degrees=45, translate=[0.25, 0.25, 0.25], scale=[0.025, 0.025, 0.025])
                ses01_subvolume, metadata = random_affine(sample=ses01_subvolume, metadata={})
                ses02_subvolume, _ = random_affine(sample=ses02_subvolume, metadata=metadata)
                gt_subvolume, _ = random_affine(sample=gt_subvolume, metadata=metadata)
            # (2.b) Apply random elastic transform with P = 0.4
            else:
                # NOTE: `metadata` ensures that the same affine is applied to all three subvolumes
                random_elastic_transform = ElasticTransform(alpha_range=(25.0, 35.0), sigma_range=(3.5, 5.5), p=1.0)
                ses01_subvolume, metadata = random_elastic_transform(sample=ses01_subvolume, metadata={})
                ses02_subvolume, _ = random_elastic_transform(sample=ses02_subvolume, metadata=metadata)
                gt_subvolume, _ = random_elastic_transform(sample=gt_subvolume, metadata=metadata)

            # (3) Apply random bias to mimic MRI bias-field artifact with P = 0.25 (independently)
            # NOTE: `torchio` expects 4D tensor with channel dim. so we unsqueeze & squeeze
            random_bias_field = Compose([RandomBiasField(coefficients=random.choice([0.1, 0.2, 0.3]), order=3, p=0.25)])
            ses01_subvolume = random_bias_field(ses01_subvolume[np.newaxis, ...])[0, :, :, :]
            random_bias_field = Compose([RandomBiasField(coefficients=random.choice([0.1, 0.2, 0.3]), order=3, p=0.25)])
            ses02_subvolume = random_bias_field(ses02_subvolume[np.newaxis, ...])[0, :, :, :]

        # Do a check on subvolume sizes after the applied augmentations
        assert ses01_subvolume.shape == ses02_subvolume.shape == gt_subvolume.shape == self.subvolume_size

        # Normalize images to zero mean and unit variance
        if ses01_subvolume.std() < 1e-5 or ses02_subvolume.std() < 1e-5:
            # If subvolumes uniform: do mean-subtraction
            ses01_subvolume = ses01_subvolume - ses01_subvolume.mean()
            ses02_subvolume = ses02_subvolume - ses02_subvolume.mean()
        else:
            normalize_instance = NormalizeInstance()
            ses01_subvolume, _ = normalize_instance(sample=ses01_subvolume, metadata={})
            ses02_subvolume, _ = normalize_instance(sample=ses02_subvolume, metadata={})

        # Extract & return patches from subvolumes (if applicable -> needed for transformer model)
        if self.use_patches and self.subvolume_size != self.patch_size:
            ses01_patches = subvolume2patches(subvolume=ses01_subvolume, patch_size=self.patch_size)
            ses02_patches = subvolume2patches(subvolume=ses02_subvolume, patch_size=self.patch_size)
            gt_patches = subvolume2patches(subvolume=gt_subvolume, patch_size=self.patch_size)

            # Compute classification GTs
            clf_gt_patches = [int(np.any(gt_patch)) for gt_patch in gt_patches]

            # Conversion to PyTorch tensors
            x1 = torch.tensor(ses01_patches, dtype=torch.float)
            x2 = torch.tensor(ses02_patches, dtype=torch.float)
            seg_y = torch.tensor(gt_patches, dtype=torch.float)
            clf_y = torch.tensor(clf_gt_patches, dtype=torch.long)
            # NOTE: The times two is added because of x1 and x2 logic in our current model

            return index, x1, x2, seg_y, clf_y

        # Return subvolumes, i.e. don't extract patches
        else:
            # Conversion to PyTorch tensors
            x1 = torch.tensor(ses01_subvolume, dtype=torch.float)
            x2 = torch.tensor(ses02_subvolume, dtype=torch.float)
            seg_y = torch.tensor(gt_subvolume, dtype=torch.float)
            clf_y = torch.tensor(int(np.any(gt_subvolume)), dtype=torch.long)

            return index, x1, x2, seg_y, clf_y

    def __len__(self):
        return len(self.subvolumes)

    def test(self, model, device, num_mc_samples):
        """Implements the test phase via animaSegPerfAnalyzer"""
        assert model is not None

        # Convert to list in case we need to assign new values in the next code block
        adjusted_center_crop_size = list(self.center_crop_size)

        # Center-crop size vs. subvolume size check to see if we need to pad the inputs or not
        for i in range(3):
            ccs_i, svs_i = adjusted_center_crop_size[i], self.subvolume_size[i]
            if ccs_i % svs_i != 0:
                # Minimally pad the initial center-crop size s.t. it becomes divisible by SV
                adjusted_center_crop_size[i] = ((ccs_i // svs_i) * svs_i) + svs_i

        # Convert back to tuple for the center-crop parameter to be immutable again
        adjusted_center_crop_size = tuple(adjusted_center_crop_size)

        # Report center-crop size parameters
        print('Original Center-Crop Size: ', self.center_crop_size)
        if adjusted_center_crop_size != self.center_crop_size:
            print('[WARNING] Adjusted Center-Crop Size: ', adjusted_center_crop_size)

        # Compute num. subvolumes per dim and total for a quick check later on
        num_subvolumes_per_dim = [adjusted_center_crop_size[i] // self.subvolume_size[i] for i in range(3)]
        num_subvolumes = np.prod(num_subvolumes_per_dim)

        for subject_no, subject in enumerate(tqdm(self.subjects_hold_out, desc='Testing Phase')):
            # Read-in input test volumes
            ses01 = nib.load(os.path.join(self.root, subject, 'anat', '%s_FLAIR.nii.gz' % subject), mmap=False)
            ses02 = nib.load(os.path.join(self.root, subject, 'anat', '%s_T1w.nii.gz' % subject), mmap=False)

            # Read-in GT volumes (using the consensus GT for now)
            gtc = nib.load(os.path.join(self.root, 'derivatives', 'labels', subject, 'anat', '%s_FLAIR_seg-lesion.nii.gz' % subject), mmap=False)

            # Check if image sizes and resolutions match
            assert ses01.shape == ses02.shape == gtc.shape
            assert ses01.header['pixdim'].tolist() == ses02.header['pixdim'].tolist() == gtc.header['pixdim'].tolist()
            assert ses01.header['pixdim'].tolist()[1:4] == [0.5, 0.5, 0.5]
            # NOTE: We know that the above voxel dimensions will hold thanks to preprocessing step

            # Convert to NumPy
            ses01, ses02, gtc = ses01.get_fdata(), ses02.get_fdata(), gtc.get_fdata()

            # Apply center-cropping
            center_crop = CenterCrop(size=adjusted_center_crop_size)
            ses01 = center_crop(sample=ses01, metadata={'crop_params': {}})[0]
            ses02 = center_crop(sample=ses02, metadata={'crop_params': {}})[0]
            gtc = center_crop(sample=gtc, metadata={'crop_params': {}})[0]

            # Get subvolumes from volumes
            # NOTE: We use subvolume size for the stride size to get non-overlapping test subvolumes
            ses01_subvolumes = volume2subvolumes(volume=ses01, subvolume_size=self.subvolume_size, stride_size=self.subvolume_size)
            ses02_subvolumes = volume2subvolumes(volume=ses02, subvolume_size=self.subvolume_size, stride_size=self.subvolume_size)
            gtc_subvolumes = volume2subvolumes(volume=gtc, subvolume_size=self.subvolume_size, stride_size=self.subvolume_size)
            assert len(ses01_subvolumes) == len(ses02_subvolumes) == len(gtc_subvolumes) == num_subvolumes

            # Collect individual subvolume predictions for full volume segmentation (i.e. full scan for one subject)
            pred_subvolumes = []
            for i in range(len(ses01_subvolumes)):
                ses01_subvolume, ses02_subvolume, gtc_subvolume = ses01_subvolumes[i], ses02_subvolumes[i], gtc_subvolumes[i]

                # Normalize images to zero mean and unit variance
                if ses01_subvolume.std() < 1e-5 or ses02_subvolume.std() < 1e-5:
                    # If subvolumes uniform: do mean-subtraction
                    ses01_subvolume = ses01_subvolume - ses01_subvolume.mean()
                    ses02_subvolume = ses02_subvolume - ses02_subvolume.mean()
                else:
                    normalize_instance = NormalizeInstance()
                    ses01_subvolume, _ = normalize_instance(sample=ses01_subvolume, metadata={})
                    ses02_subvolume, _ = normalize_instance(sample=ses02_subvolume, metadata={})

                # NOTE: Create twice-unsqueezed tensors with batch_size=1 and num_channels=1
                if self.use_patches:
                    ses01_patches = subvolume2patches(ses01_subvolume, patch_size=self.patch_size)
                    ses02_patches = subvolume2patches(ses02_subvolume, patch_size=self.patch_size)
                    x1 = torch.tensor(ses01_patches, dtype=torch.float).view(1, ses01_patches.shape[0], 1, *ses01_patches.shape[1:]).to(device)
                    x2 = torch.tensor(ses02_patches, dtype=torch.float).view(1, ses02_patches.shape[0], 1, *ses02_patches.shape[1:]).to(device)
                else:
                    x1 = torch.tensor(ses01_subvolume, dtype=torch.float).view(1, 1, *ses01_subvolume.shape).to(device)
                    x2 = torch.tensor(ses02_subvolume, dtype=torch.float).view(1, 1, *ses02_subvolume.shape).to(device)

                if num_mc_samples > 0:   # Do Monte Carlo Averaging
                    seg_y_hats = []
                    # Forward pass through the network num_mc_samples times and combine them by averaging.
                    for i_mc in range(num_mc_samples):
                        seg_y_hats.append(model(x1, x2).squeeze())
                    seg_y_hats = torch.stack(seg_y_hats)
                    seg_y_hat = torch.mean(seg_y_hats, dim=0).detach().cpu().numpy()
                else:    # Get the standard subvolume prediction
                    seg_y_hat = model(x1, x2).squeeze().detach().cpu().numpy()

                if self.use_patches:
                    seg_y_hat = patches2subvolume(patches=list(seg_y_hat), subvolume_size=self.subvolume_size)

                # Optional visualization for manual assessment of performance
                if self.visualize_test_preds:
                    if np.any(gtc_subvolume):
                        seg_y_hat_nib = nib.Nifti1Image(seg_y_hat, affine=np.eye(4))
                        nib.save(img=seg_y_hat_nib, filename=os.path.join(self.results_dir, '%s_%d_pred.nii.gz' % (subject, i)))
                        gtc_subvolume_nib = nib.Nifti1Image(gtc_subvolume, affine=np.eye(4))
                        nib.save(img=gtc_subvolume_nib, filename=os.path.join(self.results_dir, '%s_%d_gt.nii.gz' % (subject, i)))

                pred_subvolumes.append(seg_y_hat)

            # Convert the list of subvolume predictions to a single volume segmentation / prediction
            pred = subvolumes2volume(subvolumes=pred_subvolumes, volume_size=adjusted_center_crop_size)
            assert pred.shape == gtc.shape

            # Apply the original center-crop size in case it was adjusted before
            if pred.shape != self.center_crop_size:
                gtc_sum_before_crop = np.sum(gtc)
                center_crop = CenterCrop(size=self.center_crop_size)
                pred = center_crop(sample=pred, metadata={'crop_params': {}})[0]
                gtc = center_crop(sample=gtc, metadata={'crop_params': {}})[0]

                # Check if padding & un-padding removes any lesion GTs; only continue if it does not
                if abs(np.sum(gtc) - gtc_sum_before_crop) > 1e-6:
                    # NOTE: Apparently np.sum() can have epsilon differences even with same values!
                    raise ValueError('Padding & un-padding cropped out lesions! Check your center-crop parameters!')

            # Convert predictions and GT to binary to be compatible with ANIMA metrics
            pred = np.array(pred > 0.5, dtype=float)
            gtc = np.array(gtc > 0.5, dtype=float)

            # Save the prediction and the center-cropped GT as new NIfTI files
            # NOTE: We are deleting & overwriting the NIfTI files at each iteration
            pred_nib = nib.Nifti1Image(pred, affine=np.eye(4))
            gtc_nib = nib.Nifti1Image(gtc, affine=np.eye(4))
            nib.save(img=pred_nib, filename=os.path.join(self.results_dir, 'pred.nii.gz'))
            nib.save(img=gtc_nib, filename=os.path.join(self.results_dir, 'gt.nii.gz'))

            # Run ANIMA segmentation performance metrics on the predictions
            # NOTE: We use certain additional arguments below with the following purposes:
            #       -d -> surface distance eval, -l -> detection of lesions eval
            #       -a -> intra-lesion eval, -s -> segmentation eval, -X -> save as XML file
            seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -l -a -s -X'
            os.system(seg_perf_analyzer_cmd %
                      (os.path.join(self.anima_binaries_path, 'animaSegPerfAnalyzer'),
                       os.path.join(self.results_dir, 'pred.nii.gz'),
                       os.path.join(self.results_dir, 'gt.nii.gz'),
                       os.path.join(self.results_dir, subject)))

            # Delete temporary NIfTI files
            os.remove(os.path.join(self.results_dir, 'pred.nii.gz'))
            os.remove(os.path.join(self.results_dir, 'gt.nii.gz'))

        # Get all XML filepaths where ANIMA performance metrics are saved for each hold-out subject
        subject_filepaths = [os.path.join(self.results_dir, f) for f in
                             os.listdir(self.results_dir) if f.endswith('.xml')]
        test_metrics = defaultdict(list)

        # Update the test metrics dictionary by iterating over all subjects
        for subject_filepath in subject_filepaths:
            subject = os.path.split(subject_filepath)[-1].split('_')[0]
            root_node = ET.parse(source=subject_filepath).getroot()

            # Check if RelativeVolumeError is INF -> means the GT is empty and should be ignored
            rve_metric = list(root_node)[6]
            assert rve_metric.get('name') == 'RelativeVolumeError'
            if np.isinf(float(rve_metric.text)):
                print('Skipping Subject=%s ENTIRELY Due to Empty GT!' % subject)
                continue

            for metric in list(root_node):
                name, value = metric.get('name'), float(metric.text)

                if np.isinf(value) or np.isnan(value):
                    print('Skipping Metric=%s for Subject=%s Due to INF or NaNs!' % (name, subject))
                    continue

                test_metrics[name].append(value)

        # Print aggregation of each metric via mean and standard dev.
        print('Test Phase Metrics [ANIMA]: ')
        for key in test_metrics:
            print('\t%s -> Mean: %0.4f Std: %0.2f' % (key, np.mean(test_metrics[key]), np.std(test_metrics[key])))


class MSSeg1Dataset(Dataset):
    """
    Custom PyTorch dataset for the MSSeg1 Challenge 2016. Works only with 3D subvolumes. Implements
    training and validation phases. Training and validation is utilized via the canonic
    __get_item__() function and by carefully setting the `train` parameter before
    accessing an item (e.g. via iterating over the dataloader in modeling/train.py).

    # TODO: Lot's of common code with MSSeg2Dataset, make it inherit from there if time allows!

    :param (float) fraction_data: Fraction of subjects to use for the entire dataset. Helps with debugging.
    :param (tuple) resolution: The common 3D resolution to resample all subjects / volumes.
    :param (tuple) center_crop_size: The 3D center-crop size for the volumes. For now, we can
           leave this at it's default value (320, 384, 512).
    :param (tuple) subvolume_size: The 3D subvolume size to be used in training & validation.
    :param (tuple) stride_size: The 3D stride size to be used in training & validation.
    :param (tuple) patch_size: The 3D patch size to be used in training & validation. (TransUNet3D-specific)
    :param (str) gt_type: The GT to use as the target masks. Leaving it at it's default 'staple'
           seems like the best option for now.
    :param (bool) use_patches: Set to True for TransUNet3D and to False for other models.
    """
    def __init__(self, root, resolution=(0.5, 0.5, 0.5), center_crop_size=(512, 512, 512),
                 subvolume_size=(128, 128, 128), stride_size=(64, 64, 64), patch_size=(32, 32, 32),
                 fraction_data=1.0, gt_type='staple', use_patches=True):
        super(MSSeg1Dataset).__init__()

        # Quick argument checks
        if not os.path.exists(root):
            raise ValueError('Specified path=%s for the challenge data can NOT be found!' % root)

        if len(center_crop_size) != 3:
            raise ValueError('The `MSChallenge3D()` expects a 3D center crop size (e.g. 512x512x512)!')
        if len(subvolume_size) != 3:
            raise ValueError('The `MSChallenge3D()` expects a 3D subvolume size (e.g. 128x128x128)!')
        if len(stride_size) != 3:
            raise ValueError('The `MSChallenge3D()` expects a 3D stride size (e.g. 64x64x64)!')
        if len(patch_size) != 3:
            raise ValueError('The `MSChallenge3D()` expects a 3D patch size (e.g. 32x32x32)!')

        if any([center_crop_size[i] < subvolume_size[i] for i in range(3)]):
            raise ValueError('The center crop size must be >= subvolume size in all dimensions!')
        if any([(center_crop_size[i] - subvolume_size[i]) % stride_size[i] != 0 for i in range(3)]):
            raise ValueError('center_crop_size - subvolume_size % stride size is NOT 0 for all dimensions!')
        if any([subvolume_size[i] < patch_size[i] for i in range(3)]):
            raise ValueError('The subvolume size must be >= patch size in all dimensions!')

        if not 0.0 < fraction_data <= 1.0:
            raise ValueError('`fraction_data` needs to be between 0.0 and 1.0!')

        if gt_type not in ['expert1', 'expert2', 'expert3', 'expert4', 'expert5', 'expert6', 'expert7', 'random', 'staple', 'average']:
            raise ValueError('gt_type=%s not recognized!' % gt_type)

        self.center_crop_size = center_crop_size
        self.subvolume_size = subvolume_size
        self.stride_size = stride_size
        self.patch_size = patch_size
        self.use_patches = use_patches
        self.gt_type = gt_type
        self.train = False

        # Get all subjects
        subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
        subjects = subjects_df['participant_id'].values.tolist()

        # Only use subset of the dataset if applicable (used for debugging)
        if fraction_data != 1.0:
            subjects = subjects[:int(len(subjects) * fraction_data)]

        # Iterate over all subjects and extract subvolumes
        self.subvolumes, self.positive_indices = [], []
        num_negatives, num_positives = 0, 0

        for subject_no, subject in enumerate(tqdm(subjects, desc='Loading Volumes -> Preparing Subvolumes')):
            # Read-in input volumes
            ses01 = nib.load(os.path.join(root, subject, 'anat', '%s_FLAIR.nii.gz' % subject), mmap=False)

            # Read-in GT volumes
            gt = None
            if gt_type == 'staple':
                gt = nib.load(os.path.join(root, 'derivatives', 'labels', subject, 'anat', '%s_FLAIR_seg-lesion0.nii.gz' % subject), mmap=False)
            elif gt_type == 'average':
                gt = nib.load(os.path.join(root, 'derivatives', 'labels', subject, 'anat', '%s_FLAIR_seg-average-lesion.nii.gz' % subject), mmap=False)
            else:
                raise NotImplementedError('gt_type=%s is not yet implemented!')

            # Check if image sizes and resolutions match
            assert ses01.shape == gt.shape
            # assert ses01.header['pixdim'].tolist() == gt.header['pixdim'].tolist()  -> apparently this is not the case!
            ses01_original_resolution = ses01.header['pixdim'].tolist()[1:4]
            gt_original_resolution = gt.header['pixdim'].tolist()[1:4]

            # Convert to NumPy
            ses01 = ses01.get_fdata()
            gt = gt.get_fdata()

            # Apply resampling
            # NOTE: This is not required for MSSeg2 as we rely on the preprocessing step for that
            resample = Resample(hspace=resolution[0], wspace=resolution[1], dspace=resolution[2])
            ses01 = resample(sample=ses01, metadata={'zooms': ses01_original_resolution, 'data_type': 'im'})[0]
            gt = resample(sample=gt, metadata={'zooms': gt_original_resolution, 'data_type': 'gt'})[0]

            # Apply center-cropping
            center_crop = CenterCrop(size=center_crop_size)
            ses01 = center_crop(sample=ses01, metadata={'crop_params': {}})[0]
            gt = center_crop(sample=gt, metadata={'crop_params': {}})[0]

            # Get subvolumes from volumes and update the list
            ses01_subvolumes = volume2subvolumes(volume=ses01, subvolume_size=self.subvolume_size, stride_size=self.stride_size)
            gt_subvolumes = volume2subvolumes(volume=gt, subvolume_size=self.subvolume_size, stride_size=self.stride_size)

            assert len(ses01_subvolumes) == len(gt_subvolumes)

            for i in range(len(ses01_subvolumes)):
                subvolumes_ = {
                    'ses01': ses01_subvolumes[i],
                    'gt': gt_subvolumes[i],
                }

                self.subvolumes.append(subvolumes_)

                # Measure positiveness based on the GT
                if np.any(gt_subvolumes[i]):
                    self.positive_indices.append(int(subject_no * len(ses01_subvolumes) + i))
                    num_positives += 1
                else:
                    num_negatives += 1

        self.inbalance_factor = num_negatives // num_positives
        print('Factor of overall inbalance is %d!' % self.inbalance_factor)

        print('Extracted a total of %d subvolumes!' % len(self.subvolumes))

    def __getitem__(self, index):
        # Retrieve subvolumes belonging to this index
        subvolumes = self.subvolumes[index]
        ses01_subvolume, gt_subvolume = subvolumes['ses01'], subvolumes['gt']

        if self.train:
            # (1) Apply random LR (lateral / left-right) flipping (i.e. axis=0) with P = 0.5
            if random.random() < 0.5:
                ses01_subvolume = np.flip(ses01_subvolume, axis=0)
                gt_subvolume = np.flip(gt_subvolume, axis=0)

            # (2.a) Apply random affine: rotation, translation, and scaling with P = 0.6
            if random.random() < 0.6:
                # NOTE: `metadata` ensures that the same affine is applied to all three subvolumes
                random_affine = RandomAffine(degrees=45, translate=[0.25, 0.25, 0.25], scale=[0.025, 0.025, 0.025])
                ses01_subvolume, metadata = random_affine(sample=ses01_subvolume, metadata={})
                gt_subvolume, _ = random_affine(sample=gt_subvolume, metadata=metadata)
            # (2.b) Apply random elastic transform with P = 0.4
            else:
                # NOTE: `metadata` ensures that the same affine is applied to all three subvolumes
                random_elastic_transform = ElasticTransform(alpha_range=(25.0, 35.0), sigma_range=(3.5, 5.5), p=1.0)
                ses01_subvolume, metadata = random_elastic_transform(sample=ses01_subvolume, metadata={})
                gt_subvolume, _ = random_elastic_transform(sample=gt_subvolume, metadata=metadata)

            # (3) Apply random bias to mimic MRI bias-field artifact with P = 0.25
            # NOTE: `torchio` expects 4D tensor with channel dim. so we unsqueeze & squeeze
            random_bias_field = Compose([RandomBiasField(coefficients=random.choice([0.1, 0.2, 0.3]), order=3, p=0.25)])
            ses01_subvolume = random_bias_field(ses01_subvolume[np.newaxis, ...])[0, :, :, :]

        # Do a check on subvolume sizes after the applied augmentations
        assert ses01_subvolume.shape == gt_subvolume.shape == self.subvolume_size

        # Normalize images to zero mean and unit variance
        if ses01_subvolume.std() < 1e-5:
            # If subvolume uniform: do mean-subtraction
            # NOTE: This will also help with discarding empty inputs!
            ses01_subvolume = ses01_subvolume - ses01_subvolume.mean()
        else:
            normalize_instance = NormalizeInstance()
            ses01_subvolume, _ = normalize_instance(sample=ses01_subvolume, metadata={})

        # Extract & return patches from subvolume (if applicable -> needed for transformer model)
        # NOTE: We are still returning x2 (as zeros) to be model compatible!
        if self.use_patches and self.subvolume_size != self.patch_size:
            ses01_patches = subvolume2patches(subvolume=ses01_subvolume, patch_size=self.patch_size)
            gt_patches = subvolume2patches(subvolume=gt_subvolume, patch_size=self.patch_size)

            # Compute classification GTs
            clf_gt_patches = [int(np.any(gt_patch)) for gt_patch in gt_patches]

            # Conversion to PyTorch tensors
            x1 = torch.tensor(ses01_patches, dtype=torch.float)
            x2 = torch.zeros_like(x1)
            seg_y = torch.tensor(gt_patches, dtype=torch.float)
            clf_y = torch.tensor(clf_gt_patches, dtype=torch.long)
            # NOTE: The times two is added because of x1 and x2 logic in our current model

            return index, x1, x2, seg_y, clf_y

        # Return subvolume, i.e. don't extract patches
        else:
            # Conversion to PyTorch tensors
            x1 = torch.tensor(ses01_subvolume, dtype=torch.float)
            x2 = torch.zeros_like(x1)
            seg_y = torch.tensor(gt_subvolume, dtype=torch.float)
            clf_y = torch.tensor(int(np.any(gt_subvolume)), dtype=torch.long)

            return index, x1, x2, seg_y, clf_y

    def __len__(self):
        return len(self.subvolumes)
