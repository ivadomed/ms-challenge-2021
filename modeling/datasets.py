import os
import random
from tqdm import tqdm
import random
import pandas as pd
import numpy as np
import nibabel as nib

import torch
from torch.utils.data import Dataset

from ivadomed.transforms import CenterCrop, RandomAffine, NormalizeInstance


class MSSeg2Dataset(Dataset):
    """Custom PyTorch dataset for the MSSeg2 Challenge 2021. Works only with 3D subvolumes."""
    def __init__(self, root, center_crop_size=(512, 512, 512), subvolume_size=(128, 128, 128),
                 stride_size=(64, 64, 64), patch_size=(32, 32, 32),
                 fraction_data=1.0, gt_type='consensus', use_patches=True, seed=42):
        super(MSSeg2Dataset).__init__()

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
            raise ValueError('center_crop_size - subvolume_size % stride size must be 0 for all dimensions!')
        if any([subvolume_size[i] < patch_size[i] for i in range(3)]):
            raise ValueError('The subvolume size must be >= patch size in all dimensions!')

        if not 0.0 < fraction_data <= 1.0:
            raise ValueError('`fraction_data` needs to be between 0.0 and 1.0!')

        self.center_crop_size = center_crop_size
        self.subvolume_size = subvolume_size
        self.stride_size = stride_size
        self.patch_size = patch_size
        self.use_patches = use_patches
        self.train = False

        # Get all subjects
        subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
        subjects = subjects_df['participant_id'].values.tolist()

        # Only use subset of the dataset if applicable (used for debugging)
        if fraction_data != 1.0:
            subjects = subjects[:int(len(subjects) * fraction_data)]

        # Iterate over all subjects and extract subvolumes
        self.subvolumes = []
        num_negatives, num_positives = 0, 0
        for subject in tqdm(subjects, desc='Reading-in volumes'):
            # Read-in input volumes
            ses01 = nib.load(os.path.join(root, subject, 'anat', '%s_FLAIR.nii.gz' % subject))
            ses02 = nib.load(os.path.join(root, subject, 'anat', '%s_T1w.nii.gz' % subject))

            # Read-in GT volumes
            if gt_type == 'consensus':
                gt = nib.load(os.path.join(root, 'derivatives', 'labels', subject, 'anat', '%s_FLAIR_seg-lesion.nii.gz' % subject))
            else:
                expert_no = int(gt_type[-1])
                gt = nib.load(os.path.join(root, 'derivatives', 'labels', subject, 'anat', '%s_FLAIR_acq-expert%d_lesion-manual.nii.gz' % (subject, expert_no)))

            # Check if image sizes and resolutions match
            assert ses01.shape == ses02.shape == gt.shape
            assert ses01.header['pixdim'].tolist() == ses02.header['pixdim'].tolist() == gt.header['pixdim'].tolist()

            # Convert to NumPy
            ses01, ses02, gt = ses01.get_fdata(), ses02.get_fdata(), gt.get_fdata()

            # Apply center-cropping
            center_crop = CenterCrop(size=center_crop_size)
            ses01 = center_crop(sample=ses01, metadata={'crop_params': {}})[0]
            ses02 = center_crop(sample=ses02, metadata={'crop_params': {}})[0]
            gt = center_crop(sample=gt, metadata={'crop_params': {}})[0]

            # Get subvolumes from volumes and update the list
            ses01_subvolumes = self.volume2subvolumes(volume=ses01)
            ses02_subvolumes = self.volume2subvolumes(volume=ses02)
            gt_subvolumes = self.volume2subvolumes(volume=gt)
            assert len(ses01_subvolumes) == len(ses02_subvolumes) == len(gt_subvolumes)

            for i in range(len(ses01_subvolumes)):
                subvolumes_ = {
                    'ses01': ses01_subvolumes[i],
                    'ses02': ses02_subvolumes[i],
                    'gt': gt_subvolumes[i]
                }
                self.subvolumes.append(subvolumes_)

                if np.any(gt_subvolumes[i]):
                    num_positives += 1
                else:
                    num_negatives += 1

        print('Factor of inbalance is %d!' % (num_negatives // num_positives))

        # Shuffle subvolumes just in case
        random.seed(seed)
        random.shuffle(self.subvolumes)
        print('Extracted a total of %d subvolumes!' % len(self.subvolumes))

    def volume2subvolumes(self, volume):
        """Converts 3D volumes into 3D subvolumes"""
        subvolumes = []
        assert volume.ndim == 3

        for x in range(0, (volume.shape[0] - self.subvolume_size[0]) + 1, self.stride_size[0]):
            for y in range(0, (volume.shape[1] - self.subvolume_size[1]) + 1, self.stride_size[1]):
                for z in range(0, (volume.shape[2] - self.subvolume_size[2]) + 1, self.stride_size[2]):
                    subvolume = volume[x: x + self.subvolume_size[0],
                                       y: y + self.subvolume_size[1],
                                       z: z + self.subvolume_size[2]]
                    subvolumes.append(subvolume)

        return subvolumes

    def subvolume2patches(self, subvolume):
        """Extracts 3D patches from 3D subvolumes; works with PyTorch tensors."""
        patches = []
        assert subvolume.ndim == 3

        for x in range(0, (subvolume.shape[0] - self.patch_size[0]) + 1, self.patch_size[0]):
            for y in range(0, (subvolume.shape[1] - self.patch_size[1]) + 1, self.patch_size[1]):
                for z in range(0, (subvolume.shape[2] - self.patch_size[2]) + 1, self.patch_size[2]):
                    patch = subvolume[x: x + self.patch_size[0],
                                      y: y + self.patch_size[1],
                                      z: z + self.patch_size[2]]
                    patches.append(patch)

        num_patches = len(patches)
        patches = np.array(patches)
        assert patches.shape == (num_patches, *self.patch_size)

        return patches

    def __getitem__(self, index):
        # Retrieve subvolumes belonging to this index
        subvolumes = self.subvolumes[index]
        ses01_subvolumes, ses02_subvolumes, gt_subvolumes = subvolumes['ses01'], subvolumes['ses02'], subvolumes['gt']

        # Training augmentations
        if self.train:
            # Apply random affine: rotation, translation, and scaling
            # NOTE: Use of `metadata` ensures that the same affine is applied to all three subvolumes
            random_affine = RandomAffine(degrees=20, translate=[0.1, 0.1, 0.1], scale=[0.1, 0.1, 0.1])
            ses01_subvolumes, metadata = random_affine(sample=ses01_subvolumes, metadata={})
            ses02_subvolumes, _ = random_affine(sample=ses02_subvolumes, metadata=metadata)
            gt_subvolumes, _ = random_affine(sample=gt_subvolumes, metadata=metadata)

        # If subvolumes uniform: train-time -> skip to random sample, val-time -> mean-subtraction
        # NOTE: This will also help with discarding empty inputs!
        if ses01_subvolumes.std() < 1e-5 or ses02_subvolumes.std() < 1e-5:
            if self.train:
                return self.__getitem__(random.randint(0, self.__len__() - 1))
            else:
                ses01_subvolumes = ses01_subvolumes - ses01_subvolumes.mean()
                ses02_subvolumes = ses02_subvolumes - ses02_subvolumes.mean()
        # Normalize images to zero mean and unit variance
        else:
            normalize_instance = NormalizeInstance()
            ses01_subvolumes, _ = normalize_instance(sample=ses01_subvolumes, metadata={})
            ses02_subvolumes, _ = normalize_instance(sample=ses02_subvolumes, metadata={})

        # Extract & return patches from subvolumes (if applicable -> needed for transformer model)
        if self.use_patches and self.subvolume_size != self.patch_size:
            ses01_patches = self.subvolume2patches(subvolume=ses01_subvolumes)
            ses02_patches = self.subvolume2patches(subvolume=ses02_subvolumes)
            gt_patches = self.subvolume2patches(subvolume=gt_subvolumes)

            # Compute classification GTs
            clf_gt_patches = [int(np.any(gt_patch)) for gt_patch in gt_patches]

            # Conversion to PyTorch tensors
            x1 = torch.tensor(ses01_patches, dtype=torch.float)
            x2 = torch.tensor(ses02_patches, dtype=torch.float)
            seg_y = torch.tensor(gt_patches, dtype=torch.float)
            clf_y = torch.tensor(clf_gt_patches, dtype=torch.long)
            # NOTE: The times two is added because of x1 and x2 logic in our current model

            return x1, x2, seg_y, clf_y

        # Return subvolumes, i.e. don't extract patches
        else:
            # Conversion to PyTorch tensors
            x1 = torch.tensor(ses01_subvolumes, dtype=torch.float)
            x2 = torch.tensor(ses02_subvolumes, dtype=torch.float)
            seg_y = torch.tensor(gt_subvolumes, dtype=torch.float)

            return x1, x2, seg_y

    def __len__(self):
        return len(self.subvolumes)
