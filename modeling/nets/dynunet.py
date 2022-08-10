import os
import torch
from monai.networks.nets import DynUNet

def get_dynunet(kernels, strides, deep_supervision_num=3, pretrain_path=None, checkpoint=None):
    
    net = DynUNet(
        spatial_dims=3,
        in_channels=2,
        out_channels=2,
        kernel_size=kernels,
        strides=strides,
        upsample_kernel_size=strides[1:],
        norm_name="instance",
        deep_supervision=True,
        deep_supr_num=deep_supervision_num,
    )

    if checkpoint is not None:
        pretrain_path = os.path.join(pretrain_path, checkpoint)
        if os.path.exists(pretrain_path):
            net.load_state_dict(torch.load(pretrain_path))
            print("pretrained checkpoint: {} loaded".format(pretrain_path))
        else:
            print("no pretrained checkpoint")
    return net