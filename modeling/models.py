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
    """Model Config with TransUNet3D-specific hyperparameters. Infers num. patches / num. tokens."""
    def __init__(self, subvolume_size, patch_size,
                 hidden_size, mlp_dim,
                 num_layers, num_heads, attention_dropout_rate,
                 dropout_rate, layer_norm_eps,
                 head_channels,
                 device):
        self.subvolume_size = subvolume_size
        self.patch_size = patch_size
        self.hidden_size = hidden_size
        self.mlp_dim = mlp_dim
        self.num_layers = num_layers
        self.attention_dropout_rate = attention_dropout_rate
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.layer_norm_eps = layer_norm_eps
        self.head_channels = head_channels
        self.device = device

        self.num_patches = (self.subvolume_size // self.patch_size) ** 3


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


# -------------------------------- TransUNet3D Implementation --------------------------------
class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings."""
    def __init__(self, cfg):
        super(Embeddings, self).__init__()
        self.cfg = cfg

        self.feature_extractor = VideoResNet(block=BasicBlock, conv_makers=[Conv3DSimple] * 4, layers=[2, 2, 2, 2])

        self.patch_embeddings = nn.Linear(in_features=512, out_features=cfg.hidden_size)
        self.position_embeddings = nn.Embedding(cfg.num_patches, cfg.hidden_size)
        self.timepoint_embeddings = nn.Embedding(2, cfg.hidden_size)

        # self.layer_norm = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        # TODO: Should we utilize layer-norm here?

        self.dropout = nn.Dropout(cfg.dropout_rate)

    def forward(self, x, position_ids, timepoint_ids):
        # x: (B, 2 * S, 3, P, P, P)
        # position_ids: (1, 2 * S)
        # timepoint_ids: (1, 2 * S)

        # Pass all patches in all batches through feat. extractor in parallel
        x, features = self.feature_extractor(x.view(-1, 3, self.cfg.patch_size, self.cfg.patch_size, self.cfg.patch_size))
        # x: (2 * B * S, 512)
        # features: [4]
        x = F.relu(x)
        # x: (2 * B * S, 512)
        # TODO: Verify that ReLU makes sense here
        # TODO: Get features from the feature extractor

        # Return to the original shape configuration
        x = x.view(-1, 2 * self.cfg.num_patches, 512)
        # x: (B, 2 * S, 512)
        features = [features_.view(-1, 2 * self.cfg.num_patches, *features_.shape[1:]) for features_ in features]
        # features: [4]

        # Get final patch embeddings
        x = self.patch_embeddings(x)
        # x: (B, 2 * S, H)
        # TODO: TransUNet implements this more efficiently with Conv2D.

        # Get position embeddings and timepoint embeddings
        position_x = self.position_embeddings(position_ids)
        # position_x: (1, 2 * S, H)

        timepoint_x = self.timepoint_embeddings(timepoint_ids)
        # timepoint_x: (1, 2 * S, H)

        # Add all embeddings together to get to the final embedding
        embeddings = x + position_x + timepoint_x
        # embeddings: (B, 2 * S, H)

        # Apply dropout
        embeddings = self.dropout(embeddings)

        return embeddings, features


class Attention(nn.Module):
    def __init__(self, cfg):
        super(Attention, self).__init__()
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

        x, features = self.embeddings(x, position_ids, timepoint_ids)
        # x: (B, 2 * S, H)
        # features: [4]

        x, attention_weights = self.encoder(x)
        # x: (B, 2 * S, H)
        # attention_weights: [AH]

        return x, attention_weights, features


class ClassificationPredictionHead(nn.Module):
    def __init__(self, cfg):
        super(ClassificationPredictionHead, self).__init__()
        self.cfg = cfg

        self.clf = nn.Linear(cfg.hidden_size, 2)

    def forward(self, x):
        logits = self.clf(x)
        return logits


class TransUNet3D(nn.Module):
    def __init__(self, cfg):
        super(TransUNet3D, self).__init__()
        self.cfg = cfg

        self.transformer = Transformer(cfg)

        self.clf = ClassificationPredictionHead(cfg)

        self.seg = SegmentationHead(cfg)

    def forward(self, x1, x2):
        # x1: (B, S, 1, P, P, P)
        # x2: (B, S, 1, P, P, P)

        # The feature extractors require a channel of 3
        if x1.size()[2] == 1:
            x1 = x1.repeat(1, 1, 3, 1, 1, 1)
            # x1: (B, S, 3, P, P, P)
        if x2.size()[2] == 1:
            x2 = x2.repeat(1, 1, 3, 1, 1, 1)
            # x2: (B, S, 3, P, P, P)

        # Concat. TPs
        x = torch.cat([x1, x2], dim=1).to(self.cfg.device)
        # x: (B, 2 * S, 3, P, P, P)

        x, attention_weights, features = self.transformer(x)
        # x: (B, 2 * S, H)
        # attention_weights: [AH]
        # features: [4]

        clf_logits = self.clf(x[:, self.cfg.num_patches: , :])
        # clf_logits: (B, S, 2)

        """
        print('X: ', x.shape)
        for i in range(len(features)):
            print('Feat L %d: ' % i, features[i].shape)
        """

        seg_logits = self.seg(x, features)
        # seg_logits: (B, 2 * S, 32, 32, 32)

        # TODO: The below line probably doesn't make sense. Find a way to do the separation more smoothly!
        seg_logits = seg_logits[:, self.cfg.num_patches:, :, :, :]
        # seg_logits: (B, S, 32, 32, 32)

        return clf_logits, seg_logits


# -------------------------------- ResNet3D Encoder Implementation --------------------------------
class Conv3DSimple(nn.Conv3d):
    def __init__(self, in_planes, out_planes, midplanes=None, stride=1, padding=1):
        super(Conv3DSimple, self).__init__(in_channels=in_planes,
                                           out_channels=out_planes,
                                           kernel_size=(3, 3, 3),
                                           stride=stride,
                                           padding=padding,
                                           bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return stride, stride, stride


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, planes, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(planes, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicStem(nn.Module):
    """The default conv-batchnorm-relu stem"""
    def __init__(self):
        super(BasicStem, self).__init__()
        self.conv = nn.Conv3d(3, 32, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.batch_norm = nn.BatchNorm3d(32)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        return x


class VideoResNet(nn.Module):
    def __init__(self, block, conv_makers, layers, num_classes=512, zero_init_residual=False):
        """Generic ResNet video generator.

        Args:
            block (nn.Module): ResNet building block
            conv_makers (list(functions)): generator function for each layer
            layers (List[int]): number of blocks per layer
            num_classes (int, optional): Dimension of the final FC layer. Defaults to 400.
            zero_init_residual (bool, optional): Zero init bottleneck residual BN. Defaults to False.
        """
        super(VideoResNet, self).__init__()
        self.inplanes = 32

        self.stem = BasicStem()

        self.layer1 = self._make_layer(block, conv_makers[0], 32, layers[0], stride=1)
        self.layer2 = self._make_layer(block, conv_makers[1], 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, conv_makers[2], 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, conv_makers[3], 256, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(256 * block.expansion, num_classes)

        # init weights
        self._initialize_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        # Collect features from different levels -> will be passed to UNet decoder
        features = []

        x = self.stem(x)

        x = self.layer1(x)
        features.append(x)

        x = self.layer2(x)
        features.append(x)

        x = self.layer3(x)
        features.append(x)

        x = self.layer4(x)
        features.append(x)

        x = self.avgpool(x)

        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x, features

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = list()
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


# -------------------------------- UNet3D Decoder Implementation --------------------------------
class SegmentationHead(nn.Module):
    def __init__(self, cfg):
        super(SegmentationHead, self).__init__()
        self.cfg = cfg

        self.upsampler1 = nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')
        self.conv1 = nn.Conv3d(self.cfg.hidden_size + 256, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm1 = nn.InstanceNorm3d(128, momentum=0.9)

        self.upsampler2 = nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')
        self.conv2 = nn.Conv3d(256, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm2 = nn.InstanceNorm3d(64, momentum=0.9)

        self.upsampler3 = nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')
        self.conv3 = nn.Conv3d(128, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm3 = nn.InstanceNorm3d(32, momentum=0.9)

        self.upsampler4 = nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')
        self.conv4 = nn.Conv3d(64, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.inorm4 = nn.InstanceNorm3d(64, momentum=0.9)

        self.upsampler5 = nn.Upsample(scale_factor=(2, 2, 2), mode='nearest')
        self.conv5 = nn.Conv3d(16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, x, features):
        # x: (B, 2 * S, H)
        # feat0: (B, 2 * S, 32, 16, 16, 16)
        # feat1: (B, 2 * S, 64, 8, 8, 8)
        # feat2: (B, 2 * S, 128, 4, 4, 4)
        # feat3: (B, 2 * S, 256, 2, 2, 2)

        # ---- 1 ----
        x = x.view(x.size()[0] * x.size()[1], x.size()[2], 1, 1, 1)
        # x: (2 * B * S, H, 1, 1, 1)
        x = self.upsampler1(x)
        # x: (2 * B * S, H, 2, 2, 2)
        x = x.view(-1, 2 * self.cfg.num_patches, *x.size()[1:])
        # x: (B, 2 * S, H, 2, 2, 2)

        x = torch.cat([x, features[-1]], dim=2)
        # x: (B, 2 * S, H + 256, 2, 2, 2)

        x = x.view(x.size()[0] * x.size()[1], *x.size()[2:])
        # x: (2 * B * S, H + 256, 2, 2, 2)

        x = self.conv1(x)
        x = self.inorm1(x)
        x = F.leaky_relu(x)
        # x: (2 * B * S, 128, 2, 2, 2)

        # ---- 2 ----
        x = self.upsampler2(x)
        # x: (2 * B * S, 128, 4, 4, 4)
        x = x.view(-1, 2 * self.cfg.num_patches, *x.size()[1:])
        # x: (B, 2 * S, 128, 4, 4, 4)

        x = torch.cat([x, features[-2]], dim=2)
        # x: (B, 2 * S, 256, 4, 4, 4)

        x = x.view(x.size()[0] * x.size()[1], *x.size()[2:])
        # x: (2 * B * S, 256, 4, 4, 4)

        x = self.conv2(x)
        x = self.inorm2(x)
        x = F.leaky_relu(x)
        # x: (2 * B * S, 64, 4, 4, 4)

        # ---- 3 ----
        x = self.upsampler3(x)
        # x: (2 * B * S, 64, 8, 8, 8)
        x = x.view(-1, 2 * self.cfg.num_patches, *x.size()[1:])
        # x: (B, 2 * S, 64, 8, 8, 8)

        x = torch.cat([x, features[-3]], dim=2)
        # x: (B, 2 * S, 128, 8, 8, 8)

        x = x.view(x.size()[0] * x.size()[1], *x.size()[2:])
        # x: (2 * B * S, 128, 8, 8, 8)

        x = self.conv3(x)
        x = self.inorm3(x)
        x = F.leaky_relu(x)
        # x: (2 * B * S, 32, 8, 8, 8)

        # ---- 4 ----
        x = self.upsampler4(x)
        # x: (2 * B * S, 32, 16, 16, 16)
        x = x.view(-1, 2 * self.cfg.num_patches, *x.size()[1:])
        # x: (B, 2 * S, 32, 16, 16, 16)

        x = torch.cat([x, features[-4]], dim=2)
        # x: (B, 2 * S, 64, 16, 16, 16)

        x = x.view(x.size()[0] * x.size()[1], *x.size()[2:])
        # x: (2 * B * S, 64, 16, 16, 16)

        x = self.conv4(x)
        x = self.inorm4(x)
        x = F.leaky_relu(x)
        # x: (2 * B * S, 64, 16, 16, 16)

        # ---- 5 ----
        x = self.upsampler5(x)
        # x: (2 * B * S, 64, 32, 32, 32)

        x = self.conv5(x)
        # x: (2 * B * S, 1, 32, 32, 32)

        x = x.view(-1, 2 * self.cfg.num_patches, *x.size()[1:])[:, :, 0, :, :]
        # x: (B, 2 * S, 32, 32, 32)

        return x

