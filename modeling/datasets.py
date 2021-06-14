import os
from tqdm import tqdm
import random
import pandas as pd
import nibabel as nib
import torch
from torch.utils.data import Dataset

from ivadomed.transforms import CenterCrop, RandomAffine, NormalizeInstance


class MSSeg2Dataset(Dataset):
    """Custom PyTorch dataset for the MSSeg2 Challenge 2021. Works only with 3D patches."""
    def __init__(self, root, patch_size=(128, 128, 128), stride_size=(64, 64, 64),
                 center_crop_size=(512, 512, 512), fraction_data=1.0):
        super(MSSeg2Dataset).__init__()

        # Quick argument checks
        if not os.path.exists(root):
            raise ValueError('Specified path=%s for the challenge data can NOT be found!' % root)

        if len(patch_size) != 3:
            raise ValueError('The `MSChallenge3D()` expects a 3D patch size (e.g. 128x128x128)!')
        if len(stride_size) != 3:
            raise ValueError('The `MSChallenge3D()` expects a 3D stride size (e.g. 64x64x64)!')
        if len(center_crop_size) != 3:
            raise ValueError('The `MSChallenge3D()` expects a 3D center crop size (e.g. 512x512x512)!')

        if any([center_crop_size[i] < patch_size[i] for i in range(3)]):
            raise ValueError('The center crop size must be > patch size in all dimensions!')
        if any([(center_crop_size[i] - patch_size[i]) % stride_size[i] != 0 for i in range(3)]):
            raise ValueError('center_crop_size - patch_size % stride size must be 0 for all dimensions!')

        if not 0.0 < fraction_data <= 1.0:
            raise ValueError('`fraction_data` needs to be between 0.0 and 1.0!')

        self.patch_size = patch_size
        self.stride_size = stride_size
        self.center_crop_size = center_crop_size

        # Get all subjects
        subjects_df = pd.read_csv(os.path.join(root, 'participants.tsv'), sep='\t')
        subjects = subjects_df['participant_id'].values.tolist()

        # Only use subset of the dataset if applicable (used for debugging)
        if fraction_data != 1.0:
            subjects = subjects[:int(len(subjects) * fraction_data)]

        # Iterate over all subjects and extract patches
        self.patches = []
        for subject in tqdm(subjects, desc='Reading-in volumes'):
            # Read-in volumes
            ses01 = nib.load(os.path.join(root, subject, 'anat', '%s_FLAIR.nii.gz' % subject))
            ses02 = nib.load(os.path.join(root, subject, 'anat', '%s_T1w.nii.gz' % subject))
            gt = nib.load(os.path.join(root, 'derivatives', 'labels', subject, 'anat', '%s_FLAIR_seg-lesion.nii.gz' % subject))

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

            # Get patches from volumes and update the list
            ses01_patches = self.volume2patches(volume=ses01)
            ses02_patches = self.volume2patches(volume=ses02)
            gt_patches = self.volume2patches(volume=gt)
            assert len(ses01_patches) == len(ses02_patches) == len(gt_patches)

            for i in range(len(ses01_patches)):
                self.patches.append({
                    'ses01': ses01_patches[i],
                    'ses02': ses02_patches[i],
                    'gt': gt_patches[i]
                })

        print('Extracted a total of %d patches!' % len(self.patches))

    def volume2patches(self, volume):
        patches = []
        assert volume.ndim == 3

        for x in range(0, (volume.shape[0] - self.patch_size[0]) + 1, self.stride_size[0]):
            for y in range(0, (volume.shape[1] - self.patch_size[1]) + 1, self.stride_size[1]):
                for z in range(0, (volume.shape[2] - self.patch_size[2]) + 1, self.stride_size[2]):
                    patch = volume[x: x + self.patch_size[0],
                                   y: y + self.patch_size[1],
                                   z: z + self.patch_size[2]]
                    patches.append(patch)

        return patches

    def __getitem__(self, index):
        # Retrieve patches belonging to this index
        patches = self.patches[index]
        ses01_patches, ses02_patches, gt_patches = patches['ses01'], patches['ses02'], patches['gt']

        # Apply random affine: rotation, translation, and scaling
        # NOTE: The use of `metadata` ensures that the same affine is applied to all three patches
        random_affine = RandomAffine(degrees=20, translate=[0.1, 0.1, 0.1], scale=[0.1, 0.1, 0.1])
        ses01_patches, metadata = random_affine(sample=ses01_patches, metadata={})
        ses02_patches, _ = random_affine(sample=ses02_patches, metadata=metadata)
        gt_patches, _ = random_affine(sample=gt_patches, metadata=metadata)

        # If the patches are uniform, let's skip this sample and return a random one
        # NOTE: This will also help with discarding empty inputs!
        if ses01_patches.std() < 1e-5 or ses02_patches.std() < 1e-5:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # Normalize images to zero mean and unit variance
        normalize_instance = NormalizeInstance()
        ses01_patches, _ = normalize_instance(sample=ses01_patches, metadata={})
        ses02_patches, _ = normalize_instance(sample=ses02_patches, metadata={})

        # Conversion to PyTorch tensors
        x1 = torch.tensor(ses01_patches, dtype=torch.float)
        x2 = torch.tensor(ses02_patches, dtype=torch.float)
        y = torch.tensor(gt_patches, dtype=torch.float)

        return x1, x2, y

    def __len__(self):
        return len(self.patches)
