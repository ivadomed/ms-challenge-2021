"""Implements various models for the MSSeg2021 challenge."""
import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def swish(x):
    """Swish act. fn. -> smoothing fn. that nonlinearly interpolates between linear and ReLU fn."""
    return x * torch.sigmoid(x)


# Define activation functions
ACT2FN = {"gelu": F.gelu, "relu": F.relu, "swish": swish}


class ModelConfig(object):
    """
    Model configuration with model-specific parameters and hyperparameters. Used for Modified3DUNet
    and TransUNet3D for now. Computes num. patches and the output dimension of Modified3DUNet
    encoder for the TransUNet3D model.

    NOTE: We are following the volume -> subvolume -> patch analogy where volume is the initial
    input image (i.e. full scan of patient), subvolume is a piece of volume utilized by models such
    as Modified3DUNet, and patch is a piece of subvolume utilized by models such as TransUNet3D.

    :param (str) task: '1' for any-lesion segmentation task on MSSeg2016 and
                       '2' for new-lesion segmentation task on MSSeg2021 challenge.
    :param (int) subvolume_size: The size of the subvolume, denoted as SV.
    :param (int) patch_size: The size of the patch, denoted as P. (TransUNet3D-specific)
    :param (int) hidden_size: Hidden dim. size of transformer embeddings and outputs, denoted as H. (TransUNet3D-specific)
    :param (int) mlp_dim: Hidden dim. size of transformer block. (TransUNet3D-specific)
    :param (int) num_layers: Num. of transformer blocks that constitute the encoder, denoted as L. (TransUNet3D-specific)
    :param (int) num_heads: Num. attention heads for multi-head attention mechanism, denoted as AH. (TransUNet3D-specific)
    :param (float) attention_dropout_rate: Dropout rate for attention layers. (TransUNet3D-specific)
    :param (float) dropout_rate: Dropout rate for transformer embeddings and outputs. (TransUNet3D-specific)
    :param (float) layer_norm_eps: Epsilon for layer norm in transformer. (TransUNet3D-specific)
    :param (bool) aux_clf_task: Set True to include the auxiliary subvolume / patch classification task.
           This task tries to classify subvolumes / patches with 1 if lesion is present or 0 if not.
    :param (int) base_n_filter: Base number of filters for the ModifiedUNet3D encoder and decoders, denoted as F.
    :param (torch.device) device: Device on which to place the tensors.
    """
    def __init__(self, task, subvolume_size, patch_size, hidden_size, mlp_dim, num_layers,
                 num_heads, attention_dropout_rate, dropout_rate, layer_norm_eps, aux_clf_task,
                 base_n_filter, device):
        self.task = task
        self.subvolume_size = subvolume_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.attention_dropout_rate = attention_dropout_rate
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.aux_clf_task = aux_clf_task
        self.base_n_filter = base_n_filter
        self.device = device

        self.num_patches = (self.subvolume_size // self.patch_size) ** 3
        self.unet_encoder_out_dim = base_n_filter * 8 * ((self.patch_size // 8) ** 3)

        print('Num. Patches: ', self.num_patches)
        print('UNet3D Encoder Output Dim.: ', self.unet_encoder_out_dim)


# ---------------------------- Test Model Implementation -----------------------------
class TestModel(nn.Module):
    """Simple test model with 3D convolutions and up-convolutions, to-be used for debugging."""
    def __init__(self):
        super(TestModel, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3))
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3))

        self.upconv1 = nn.ConvTranspose3d(in_channels=16, out_channels=8, kernel_size=(3, 3, 3))
        self.upconv2 = nn.ConvTranspose3d(in_channels=8, out_channels=1, kernel_size=(3, 3, 3))

    def forward(self, x1, x2):
        # Quick check: only work with 3D volumes (extra dimension comes from batch size)
        assert len(x1.size()) == 4 and len(x2.size()) == 4

        # Expand dims (i.e. add channel dim. to beginning)
        x1_, x2_ = x1.unsqueeze(1), x2.unsqueeze(1)

        # Pass to network
        x1_, x2_ = F.relu(self.conv1(x1_)), F.relu(self.conv1(x2_))
        x1_, x2_ = F.relu(self.conv2(x1_)), F.relu(self.conv2(x2_))
        x1_, x2_ = F.relu(self.upconv1(x1_)), F.relu(self.upconv1(x2_))
        x1_, x2_ = self.upconv2(x1_), self.upconv2(x2_)

        # Simply get the differentiable diff. between the two feature maps for now
        y_hat = torch.sigmoid(x2_ - x1_)[:, 0]

        return y_hat


# ---------------------------- Helpers Implementation -----------------------------
def seq2batch(x):
    """Converts 6D tensor of size [B, S, C, P, P, P] to 5D tensor of size [B * S, C, P, P, P]"""
    return x.view(x.size(0) * x.size(1), *x.size()[2:])


def batch2seq(x, num_patches):
    """Converts 5D tensor of size [B * S, C, P, P, P] to 6D tensor of size [B, S, C, P, P, P]"""
    return x.view(-1, num_patches, *x.size()[1:])


def conv_norm_lrelu(feat_in, feat_out):
    """Conv3D + InstanceNorm3D + LeakyReLU block"""
    return nn.Sequential(
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(feat_out),
        nn.LeakyReLU()
    )


def norm_lrelu_conv(feat_in, feat_out):
    """InstanceNorm3D + LeakyReLU + Conv3D block"""
    return nn.Sequential(
        nn.InstanceNorm3d(feat_in),
        nn.LeakyReLU(),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False)
    )


def lrelu_conv(feat_in, feat_out):
    """LeakyReLU + Conv3D block"""
    return nn.Sequential(
        nn.LeakyReLU(),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False)
    )


def norm_lrelu_upscale_conv_norm_lrelu(feat_in, feat_out):
    """InstanceNorm3D + LeakyReLU + 2X Upsample + Conv3D + InstanceNorm3D + LeakyReLU block"""
    return nn.Sequential(
        nn.InstanceNorm3d(feat_in),
        nn.LeakyReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv3d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
        nn.InstanceNorm3d(feat_out),
        nn.LeakyReLU()
    )


# ---------------------------- ModifiedUNet3D Encoder Implementation -----------------------------
class ModifiedUNet3DEncoder(nn.Module):
    """Encoder for ModifiedUNet3D. Adapted from ivadomed.models"""
    def __init__(self, cfg, in_channels=1, base_n_filter=8, flatten=True):
        super(ModifiedUNet3DEncoder, self).__init__()
        self.cfg = cfg

        self.flatten = flatten

        # Initialize common operations
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Level 1 context pathway
        self.conv3d_c1_1 = nn.Conv3d(in_channels, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3d_c1_2 = nn.Conv3d(base_n_filter, base_n_filter, kernel_size=3, stride=1, padding=1, bias=False)
        self.lrelu_conv_c1 = lrelu_conv(base_n_filter, base_n_filter)
        self.inorm3d_c1 = nn.InstanceNorm3d(base_n_filter)

        # Level 2 context pathway
        self.conv3d_c2 = nn.Conv3d(base_n_filter, base_n_filter * 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c2 = norm_lrelu_conv(base_n_filter * 2, base_n_filter * 2)
        self.inorm3d_c2 = nn.InstanceNorm3d(base_n_filter * 2)

        # Level 3 context pathway
        self.conv3d_c3 = nn.Conv3d(base_n_filter * 2, base_n_filter * 4, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c3 = norm_lrelu_conv(base_n_filter * 4, base_n_filter * 4)
        self.inorm3d_c3 = nn.InstanceNorm3d(base_n_filter * 4)

        # Level 4 context pathway
        self.conv3d_c4 = nn.Conv3d(base_n_filter * 4, base_n_filter * 8, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c4 = norm_lrelu_conv(base_n_filter * 8, base_n_filter * 8)
        self.inorm3d_c4 = nn.InstanceNorm3d(base_n_filter * 8)

        # Level 5 context pathway, level 0 localization pathway
        self.conv3d_c5 = nn.Conv3d(base_n_filter * 8, base_n_filter * 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm_lrelu_conv_c5 = norm_lrelu_conv(base_n_filter * 16, base_n_filter * 16)
        self.norm_lrelu_upscale_conv_norm_lrelu_l0 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 16, base_n_filter * 8)

        if self.flatten:
            self.fc = nn.Linear(self.cfg.unet_encoder_out_dim, self.cfg.unet_encoder_out_dim)

    def forward(self, x):
        #  Level 1 context pathway
        out = self.conv3d_c1_1(x)
        residual_1 = out
        out = self.lrelu(out)
        out = self.conv3d_c1_2(out)
        out = self.dropout3d(out)
        out = self.lrelu_conv_c1(out)

        # Element Wise Summation
        out += residual_1
        context_1 = self.lrelu(out)
        out = self.inorm3d_c1(out)
        out = self.lrelu(out)

        # Level 2 context pathway
        out = self.conv3d_c2(out)
        residual_2 = out
        out = self.norm_lrelu_conv_c2(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c2(out)
        out += residual_2
        out = self.inorm3d_c2(out)
        out = self.lrelu(out)
        context_2 = out

        # Level 3 context pathway
        out = self.conv3d_c3(out)
        residual_3 = out
        out = self.norm_lrelu_conv_c3(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c3(out)
        out += residual_3
        out = self.inorm3d_c3(out)
        out = self.lrelu(out)
        context_3 = out

        # Level 4 context pathway
        out = self.conv3d_c4(out)
        residual_4 = out
        out = self.norm_lrelu_conv_c4(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c4(out)
        out += residual_4
        out = self.inorm3d_c4(out)
        out = self.lrelu(out)
        context_4 = out

        # Level 5
        out = self.conv3d_c5(out)
        residual_5 = out
        out = self.norm_lrelu_conv_c5(out)
        out = self.dropout3d(out)
        out = self.norm_lrelu_conv_c5(out)
        out += residual_5
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l0(out)

        context_features = [context_1, context_2, context_3, context_4]

        # TODO: Think about this section more, e.g. do we need act. fn. after the self.fc call?
        # NOTE: Flatten is only active for TransUNet3D, not for ModifiedUNet3D.
        if self.flatten:
            context_out = out
            out = out.flatten(start_dim=1)
            out = self.lrelu(self.fc(out))

            return out, context_out, context_features

        return out, context_features


# ---------------------------- ModifiedUNet3D Decoder Implementation -----------------------------
class ModifiedUNet3DDecoder(nn.Module):
    """Decoder for ModifiedUNet3D. Adapted from ivadomed.models"""
    def __init__(self, cfg, n_classes=1, base_n_filter=8):
        super(ModifiedUNet3DDecoder, self).__init__()
        self.cfg = cfg

        # Initialize common operations
        self.lrelu = nn.LeakyReLU()
        self.dropout3d = nn.Dropout3d(p=0.5)
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.conv3d_l0 = nn.Conv3d(base_n_filter * 8, base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.inorm3d_l0 = nn.InstanceNorm3d(base_n_filter * 8)

        # Level 1 localization pathway
        self.conv_norm_lrelu_l1 = conv_norm_lrelu(base_n_filter * 16, base_n_filter * 16)
        self.conv3d_l1 = nn.Conv3d(base_n_filter * 16, base_n_filter * 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l1 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 8, base_n_filter * 4)

        # Level 2 localization pathway
        self.conv_norm_lrelu_l2 = conv_norm_lrelu(base_n_filter * 8, base_n_filter * 8)
        self.conv3d_l2 = nn.Conv3d(base_n_filter * 8, base_n_filter * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l2 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 4, base_n_filter * 2)

        # Level 3 localization pathway
        self.conv_norm_lrelu_l3 = conv_norm_lrelu(base_n_filter * 4, base_n_filter * 4)
        self.conv3d_l3 = nn.Conv3d(base_n_filter * 4, base_n_filter * 2, kernel_size=1, stride=1, padding=0, bias=False)
        self.norm_lrelu_upscale_conv_norm_lrelu_l3 = norm_lrelu_upscale_conv_norm_lrelu(base_n_filter * 2, base_n_filter)

        # Level 4 localization pathway
        self.conv_norm_lrelu_l4 = conv_norm_lrelu(base_n_filter * 2, base_n_filter * 2)
        self.conv3d_l4 = nn.Conv3d(base_n_filter * 2, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

        self.ds2_1x1_conv3d = nn.Conv3d(base_n_filter * 8, n_classes, kernel_size=1, stride=1, padding=0, bias=False)
        self.ds3_1x1_conv3d = nn.Conv3d(base_n_filter * 4, n_classes, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x, context_features):
        # Get context features from the encoder
        context_1, context_2, context_3, context_4 = context_features

        out = self.conv3d_l0(x)
        out = self.inorm3d_l0(out)
        out = self.lrelu(out)

        # Level 1 localization pathway
        out = torch.cat([out, context_4], dim=1)
        out = self.conv_norm_lrelu_l1(out)
        out = self.conv3d_l1(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l1(out)

        # Level 2 localization pathway
        out = torch.cat([out, context_3], dim=1)
        out = self.conv_norm_lrelu_l2(out)
        ds2 = out
        out = self.conv3d_l2(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l2(out)

        # Level 3 localization pathway
        out = torch.cat([out, context_2], dim=1)
        out = self.conv_norm_lrelu_l3(out)
        ds3 = out
        out = self.conv3d_l3(out)
        out = self.norm_lrelu_upscale_conv_norm_lrelu_l3(out)

        # Level 4 localization pathway
        out = torch.cat([out, context_1], dim=1)
        out = self.conv_norm_lrelu_l4(out)
        out_pred = self.conv3d_l4(out)

        ds2_1x1_conv = self.ds2_1x1_conv3d(ds2)
        ds1_ds2_sum_upscale = self.upsample(ds2_1x1_conv)
        ds3_1x1_conv = self.ds3_1x1_conv3d(ds3)
        ds1_ds2_sum_upscale_ds3_sum = ds1_ds2_sum_upscale + ds3_1x1_conv
        ds1_ds2_sum_upscale_ds3_sum_upscale = self.upsample(ds1_ds2_sum_upscale_ds3_sum)

        out = out_pred + ds1_ds2_sum_upscale_ds3_sum_upscale

        # Final Activation Layer
        out = F.relu(out) / F.relu(out).max() if bool(F.relu(out).max()) else F.relu(out)
        out = out.squeeze()

        return out


# ---------------------------- ModifiedUNet3D Implementation -----------------------------
class ModifiedUNet3D(nn.Module):
    """ModifiedUNet3D with Encoder + Decoder. Adapted from ivadomed.models"""
    def __init__(self, cfg):
        super(ModifiedUNet3D, self).__init__()
        self.cfg = cfg

        self.unet_encoder = ModifiedUNet3DEncoder(cfg, in_channels=1 if cfg.task == '1' else 2, base_n_filter=cfg.base_n_filter, flatten=False)

        if self.cfg.aux_clf_task:
            self.clf = ClassificationPredictionHead(cfg)

        self.unet_decoder = ModifiedUNet3DDecoder(cfg, n_classes=1, base_n_filter=cfg.base_n_filter)

    def forward(self, x1, x2):
        # x1: (B, 1, SV, SV, SV)
        # x2: (B, 1, SV, SV, SV)

        if self.cfg.task == '2':
            # Concat. TPs
            x = torch.cat([x1, x2], dim=1).to(self.cfg.device)
            # x: (B, 2, SV, SV, SV)
        else:
            # Discard x2
            x = x1
            # x: (B, 1, SV, SV, SV)

        x, context_features = self.unet_encoder(x)
        # x: (B, 8 * F, SV // 8, SV // 8, SV // 8)
        # context_features: [4]
        #   0 -> (B, F, SV, SV, SV)
        #   1 -> (B, 2 * F, SV / 2, SV / 2, SV / 2)
        #   2 -> (B, 4 * F, SV / 4, SV / 4, SV / 4)
        #   3 -> (B, 8 * F, SV / 8, SV / 8, SV / 8)

        seg_logits = self.unet_decoder(x, context_features)

        return seg_logits


# -------------------------------- TransUNet3D Implementation --------------------------------
class Embeddings(nn.Module):
    """Transformer Embeddings. Constructs (i) patch, (ii) position, and (iii) timepoint embeddings."""
    def __init__(self, cfg):
        super(Embeddings, self).__init__()
        self.cfg = cfg

        self.unet_encoder = ModifiedUNet3DEncoder(cfg, in_channels=1, base_n_filter=cfg.base_n_filter, flatten=True)

        self.patch_embeddings = nn.Linear(in_features=cfg.unet_encoder_out_dim, out_features=cfg.hidden_size)
        self.position_embeddings = nn.Embedding(cfg.num_patches, cfg.hidden_size)
        self.timepoint_embeddings = nn.Embedding(2, cfg.hidden_size)

        self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)

        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, x, position_ids, timepoint_ids):
        # x: (B, 2 * S, 1, P, P, P)
        # position_ids: (1, 2 * S)
        # timepoint_ids: (1, 2 * S)

        x = seq2batch(x)
        # x: (2 * B * S, 1, P, P, P)

        x, context_out, context_features = self.unet_encoder(x)
        # x: (2 * B * S, 8 * F * (P / 8) * (P / 8) * (P / 8))
        # context_out: (2 * B * S, 8 * F, P / 8, P / 8, P / 8)
        # context_features: [4]
        #   0 -> (2 * B * S, F, P, P, P)
        #   1 -> (2 * B * S, 2 * F, P / 2, P / 2, P / 2)
        #   2 -> (2 * B * S, 4 * F, P / 4, P / 4, P / 4)
        #   3 -> (2 * B * S, 8 * F, P / 8, P / 8, P / 8)

        # Return to the original shape configuration
        x = batch2seq(x, num_patches=2 * self.cfg.num_patches)
        # x: (B, 2 * S, 8 * F * (P / 8) * (P / 8) * (P / 8))

        # Get final patch embeddings
        x = self.patch_embeddings(x)
        # x: (B, 2 * S, H)

        # Get position embeddings and timepoint embeddings
        position_x = self.position_embeddings(position_ids)
        # position_x: (1, 2 * S, H)

        timepoint_x = self.timepoint_embeddings(timepoint_ids)
        # timepoint_x: (1, 2 * S, H)
        # NOTE: With these embeddings, we would like to differentiate between TP1 and TP2.

        # Add all embeddings together to get to the final embedding
        embeddings = x + position_x + timepoint_x
        # embeddings: (B, 2 * S, H)

        # Apply layer norm. and dropout
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings, context_out, context_features


class Attention(nn.Module):
    """Multi-Head Attention"""
    def __init__(self, cfg):
        super(Attention, self).__init__()
        self.cfg = cfg

        self.num_attention_heads = cfg.num_heads
        self.attention_head_size = int(cfg.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.key = nn.Linear(cfg.hidden_size, self.all_head_size)
        self.value = nn.Linear(cfg.hidden_size, self.all_head_size)

        self.out = nn.Linear(cfg.hidden_size, cfg.hidden_size)
        self.attn_dropout = nn.Dropout(cfg.attention_dropout_rate)
        self.proj_dropout = nn.Dropout(cfg.attention_dropout_rate)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Optional masking: we want to mask empty / zero TP2 inputs for any-lesion segmentation task
        if self.cfg.task == '1':
            attention_mask = torch.ones_like(attention_scores)
            attention_mask[:, :, self.cfg.num_patches:, :] = 0
            attention_mask[:, :, :, self.cfg.num_patches:] = 0
            attention_scores = attention_scores.masked_fill(attention_mask == 0, -1e-9)
            # TODO: Make sure this is OK! Reference: https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec

        attention_probs = self.softmax(attention_scores)
        weights = attention_probs
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class MLP(nn.Module):
    """Multi-Layer Perceptron"""
    def __init__(self, cfg):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(cfg.hidden_size, cfg.mlp_dim)
        self.fc2 = nn.Linear(cfg.mlp_dim, cfg.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(cfg.dropout_rate)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    """Transformer Block with MLP + Attention + LayerNorm"""
    def __init__(self, cfg):
        super(Block, self).__init__()
        self.cfg = cfg
        self.attention_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.ffn = MLP(cfg)
        self.attn = Attention(cfg)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights


class Encoder(nn.Module):
    """Transformer Encoder with L Blocks"""
    def __init__(self, cfg):
        super(Encoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        for _ in range(cfg.num_layers):
            layer = Block(cfg)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attention_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            attention_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attention_weights


class Transformer(nn.Module):
    """Transformer: Embeddings + Encoder"""
    def __init__(self, cfg):
        super(Transformer, self).__init__()
        self.cfg = cfg
        self.embeddings = Embeddings(cfg)
        self.encoder = Encoder(cfg)

    def forward(self, x):
        # Get position IDs and timepoint IDs
        position_ids = torch.cat([torch.arange(self.cfg.num_patches)] * 2).unsqueeze(0).to(self.cfg.device)
        # position_ids: (1, 2 * S)
        timepoint_ids = torch.tensor([0] * self.cfg.num_patches + [1] * self.cfg.num_patches).unsqueeze(0).to(self.cfg.device)
        # timepoint_ids: (1, 2 * S)

        x, context_out, context_features = self.embeddings(x, position_ids, timepoint_ids)
        # x: (B, 2 * S, H)
        # context_features: [4]
        #   0 -> (2 * B * S, F, P, P, P)
        #   1 -> (2 * B * S, 2 * F, P / 2, P / 2, P / 2)
        #   2 -> (2 * B * S, 4 * F, P / 4, P / 4, P / 4)
        #   3 -> (2 * B * S, 8 * F, P / 8, P / 8, P / 8)

        x, attention_weights = self.encoder(x)
        # x: (B, 2 * S, H)
        # attention_weights: [AH] -> (B, B, 2 * S, 2 * S)

        return x, attention_weights, context_out, context_features


class ClassificationPredictionHead(nn.Module):
    """Classification Prediction Head for the auxiliary classification task."""
    def __init__(self, cfg):
        super(ClassificationPredictionHead, self).__init__()
        self.cfg = cfg

        self.clf = nn.Linear(cfg.hidden_size, 2)

    def forward(self, x):
        logits = self.clf(x)
        return logits


class TransUNet3D(nn.Module):
    """TransUNet3D implementation with Modified3DUNet + Transformer"""
    def __init__(self, cfg):
        super(TransUNet3D, self).__init__()
        self.cfg = cfg

        self.transformer = Transformer(cfg)

        if self.cfg.aux_clf_task:
            self.clf = ClassificationPredictionHead(cfg)

        self.unet_decoder = ModifiedUNet3DDecoder(cfg, n_classes=1, base_n_filter=cfg.base_n_filter)

    def forward(self, x1, x2):
        # x1: (B, S, 1, P, P, P)
        # x2: (B, S, 1, P, P, P)

        # Concat. TPs
        x = torch.cat([x1, x2], dim=1).to(self.cfg.device)
        # x: (B, 2 * S, 1, P, P, P)

        x, attention_weights, context_out, context_features = self.transformer(x)
        # x: (B, 2 * S, H)
        # context_out: (2 * B * S, 8 * F, P // 8, P // 8, P // 8)
        # attention_weights: [AH] -> (B, B, 2 * S, 2 * S)
        # context_features: [4]
        #   0 -> (2 * B * S, F, P, P, P)
        #   1 -> (2 * B * S, 2 * F, P / 2, P / 2, P / 2)
        #   2 -> (2 * B * S, 4 * F, P / 4, P / 4, P / 4)
        #   3 -> (2 * B * S, 8 * F, P / 8, P / 8, P / 8)

        # Save hidden output of transformer later for the classification task
        x_hidden = x

        x = x.view(x.size(0), x.size(1), self.cfg.base_n_filter * 8, *((self.cfg.patch_size // 8, ) * 3))
        # x: (B, 2 * S, 8 * F, P // 8, P // 8, P // 8)
        # NOTE: This works when mlp_dim == unet_encoder_output_dim -> make sure this is the case!

        x = seq2batch(x)
        # x: (2 * B * S, 8 * F, P // 8, P // 8, P // 8)

        # TODO: We can also only send the latter half to the decoder! Decide on what to do here!
        seg_logits_x1x2 = self.unet_decoder(x + context_out, context_features)
        # seg_logits_x1x2: (2 * B * S, P, P, P)
        seg_logits_x1x2 = batch2seq(seg_logits_x1x2, num_patches=2 * self.cfg.num_patches)
        # seg_logits_x1x2: (B, 2 * S, P, P, P)

        seg_logits = seg_logits_x1x2[:, :self.cfg.num_patches if self.cfg.task == '1' else self.cfg.num_patches:, :, :, :]
        # seg_logits: (B, S, P, P, P)

        if self.cfg.aux_clf_task:
            clf_logits = self.clf(x_hidden[:, :self.cfg.num_patches if self.cfg.task == '1' else self.cfg.num_patches:, :])
            # clf_logits: (B, S, 2)

            return clf_logits, seg_logits

        return seg_logits
