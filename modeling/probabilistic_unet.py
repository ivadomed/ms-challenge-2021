# Code adapted from https://github.com/stefanknegt/Probabilistic-Unet-Pytorch/blob/master/probabilistic_unet.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Independent, kl

from unet import UpConvBlock, DownConvBlock
from unet import Unet
from utils import init_weights, init_weights_orthogonal_normal, l2_regularization

from ivadomed.losses import DiceLoss


class Encoder(nn.Module):
    """
    A simple CNN consisting of convs_per_block convolutional layers + Relu activations per block for len(num_feat_maps)
    blocks with pooling between consecutive blocks
    """
    def __init__(self, in_channels, num_feat_maps, convs_per_block, initializers, padding=True, posterior=False):
        super(Encoder, self).__init__()
        self.contracting_path = nn.ModuleList()
        self.in_channels = in_channels
        self.num_feat_maps = num_feat_maps

        if posterior:
            # Recall the posterior net is conditioned upon the GT so 1 additional input channel
            self.in_channels += 1

        layers = []
        for i in range(len(self.num_feat_maps)):
            in_dim = self.in_channels if i == 0 else out_dim
            out_dim = num_feat_maps[i]
            if i != 0:
                layers.append(nn.AvgPool3d(kernel_size=3, stride=2, padding=0, ceil_mode=True))

            layers.append(nn.Conv3d(in_channels=in_dim, out_channels=out_dim, kernel_size=3, padding=int(padding)))
            layers.append(nn.ReLU(inplace=True))
            for _ in range(convs_per_block-1):
                layers.append(nn.Conv3d(in_channels=out_dim, out_channels=out_dim, kernel_size=3, padding=int(padding)))
                layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)
        self.layers.apply(init_weights)       # for kaiming_normal initialization
        # self.layers.apply(init_weights_orthogonal_normal)   # for orthogonal weight initialization

    def forward(self, x):
        output = self.layers(x)
        # print("Reached: ", output.shape)
        return output


class AxisAlignedConvGaussian(nn.Module):
    """
    A ConvNet that parameterizes a Gaussian distribution with axis-aligned covariance matrix
    """
    def __init__(self, in_channels, num_feat_maps, convs_per_block, latent_dim, initializers, posterior=False):
        super(AxisAlignedConvGaussian, self).__init__()
        self.in_channels = in_channels
        self.channel_axis = 1
        self.num_feat_maps = num_feat_maps
        self.convs_per_block = convs_per_block
        self.latent_dim = latent_dim
        self.posterior = posterior
        if self.posterior:
            self.name = "Posterior"
        else:
            self.name = "Prior"
        self.encoder = Encoder(self.in_channels, self.num_feat_maps, self.convs_per_block, initializers, self.posterior)
        self.conv_layer = nn.Conv3d(num_feat_maps[-1], 2*self.latent_dim, kernel_size=1, stride=1)
        self.show_img = 0
        self.show_seg = 0
        self.show_concat = 0
        self.show_enc = 0
        self.sum_input = 0

        nn.init.kaiming_normal_(self.conv_layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.conv_layer.bias)
        # using the orthogonal weight initialization
        # self.conv_layer.apply(init_weights_orthogonal_normal)

    def forward(self, inp, seg=None):
        # If segmentation is not None, then concatenate it to the channel axis of the input
        if seg is not None:
            self.show_img = inp
            self.show_seg = seg
            # print("input shape: ", inp.shape); print("seg mask shape: ", seg.shape)
            inp = torch.cat([inp, seg], dim=1)
            # print("concatenated input and GT shape", inp.shape)
            self.show_concat = inp
            self.sum_input = torch.sum(inp)

        encoding = self.encoder(inp)
        self.show_enc = encoding
        # for getting the mean of the resulting volume --> (batch-size x Channels x Depth x Height x Width)
        encoding = torch.mean(encoding, dim=2, keepdim=True)
        encoding = torch.mean(encoding, dim=3, keepdim=True)
        encoding = torch.mean(encoding, dim=4, keepdim=True)

        # Convert the encoding into 2xlatent_dim in the output_channels and split into mu and log_sigma
        mu_log_sigma = self.conv_layer(encoding)    # shape: (B x (2*latent_dim) x 1 x 1 x 1)

        # Squeeze all the singleton dimensions
        mu_log_sigma = torch.squeeze(mu_log_sigma)   # shape: (B x (2*latent_dim) )

        mu = mu_log_sigma[:, :self.latent_dim]      # take the first "latent_dim" samples as mu
        log_sigma = mu_log_sigma[:, self.latent_dim:]   # take the remaining as log_sigma

        # This is the multivariate normal with diagonal covariance matrix
        # https://github.com/pytorch/pytorch/pull/11178 (see below comments)
        dist = Independent(Normal(loc=mu, scale=torch.exp(log_sigma)), 1)
        return dist


class FComb(nn.Module):
    """
    As in the paper, this class creates convs_fcomb number of 1x1 conv. layers that combines the random sample taken
    from the latent space and concatenates it with the output of the U-Net (along the channel axis) to get
    the final prediction mask.
    """
    def __init__(self, num_feat_maps, latent_dim, num_out_channels, num_classes, convs_fcomb,
                 initializers, use_tile=True):
        super(FComb, self).__init__()
        self.num_out_channels = num_out_channels
        self.num_classes = num_classes
        self.channel_axis = 1
        # self.spatial_axes = [2, 3]   # 2,3? for images' H x W dimensions
        self.spatial_axes = [2, 3, 4]   # for volumes' D x H X W dimensions
        self.num_feat_maps = num_feat_maps
        self.latent_dim = latent_dim
        self.use_tile = use_tile
        self.convs_fcomb = convs_fcomb
        self.name = "FComb"

        if self.use_tile:
            # creating a small decoder containing N - (1x1 Conv + ReLU) blocks except for the last layer
            layers = [
                nn.Conv3d(self.num_feat_maps[0] + self.latent_dim, self.num_feat_maps[0], kernel_size=1),
                nn.ReLU(inplace=True)
            ]

            for _ in range(convs_fcomb-2):
                layers.append(nn.Conv3d(self.num_feat_maps[0], self.num_feat_maps[0], kernel_size=1))
                layers.append(nn.ReLU(inplace=True))

            self.layers = nn.Sequential(*layers)
            self.last_layer = nn.Conv3d(self.num_feat_maps[0], self.num_classes, kernel_size=1)

            if initializers['w'] == 'orthogonal':
                self.layers.apply(init_weights_orthogonal_normal)
                self.last_layer.apply(init_weights_orthogonal_normal)
            else:
                self.layers.apply(init_weights)
                self.last_layer.apply(init_weights)

    def tile(self, a, dim, n_tile):
        """
        This function is taken form PyTorch forum and mimics the behavior of tf.tile.
        Source: https://discuss.pytorch.org/t/how-to-tile-a-tensor/13853/3
        """
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*repeat_idx)
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).cuda()
        # order_index = torch.LongTensor(torch.cat([init_dim * torch.arange(n_tile) + i for i in range(init_dim)]))  # .to(device)
        return torch.index_select(a, dim, order_index)

    def forward(self, feature_map, z):
        """
        Z is (batch_size x latent_dim) and feature_map is (batch_size x num_channels x P x P x P)
        Z is broadcasted to (batch_size(B) x latent_dim(LD) x P x P x P). (Just like tensorflow's tf.tile function)
        """
        if self.use_tile:
            z = torch.unsqueeze(z, dim=2)       # shape: (B x LD x 1)
            z = self.tile(z, dim=2, n_tile=feature_map.shape[self.spatial_axes[0]])     # shape: (B x LD x P)
            z = torch.unsqueeze(z, dim=3)       # shape: (B x LD x P x 1)
            z = self.tile(z, dim=3, n_tile=feature_map.shape[self.spatial_axes[1]])     # shape: (B x LD x P x P)
            z = torch.unsqueeze(z, dim=4)       # shape: (B x LD x P x P x 1)
            z = self.tile(z, dim=4, n_tile=feature_map.shape[self.spatial_axes[2]])     # shape: (B x LD x P x P x P)

            # Concatenate UNet's output feature map and a sample taken from latent space
            feature_map = torch.cat((feature_map, z), dim=self.channel_axis)
            output = self.layers(feature_map)
            return self.last_layer(output)


class ProbabilisticUnet(nn.Module):
    """
    An implementation of the probabilistic U-net
    in_channels = number of channels in the input image (1 greyscale and 3 rgb) (int)
    num_classes = number of output classes to predict (int)
    num_feat_maps = list of number of feature maps (filters) per layer/resolution
    latent_dim = dimension of the latent space (int)
    convs_fcomb = number of (1x1) convolutional layers per block combining the latent space sample to U-Net's feat. map
    beta = weighting parameter for the cross-entropy loss and KL divergence.
    """
    def __init__(self, in_channels=2, num_classes=1, num_feat_maps=[32, 64, 128, 192], latent_dim=6,
                 convs_fcomb=4, beta=10.0):
        super(ProbabilisticUnet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_feat_maps = num_feat_maps
        self.latent_dim = latent_dim
        self.convs_per_block = 4
        self.convs_fcomb = convs_fcomb
        self.initializers = {'w': 'he_normal', 'b': 'normal'}
        # self.initializers = {'w': 'orthogonal', 'b': 'normal'}   # orthogonal weight initialization
        self.beta = beta
        self.z_prior_sample = 0

        # Instantiating the networks
        self.unet = Unet(self.in_channels, self.num_classes, self.num_feat_maps, self.initializers, padding=True)
        self.prior = AxisAlignedConvGaussian(self.in_channels, self.num_feat_maps, self.convs_per_block,
                                             self.latent_dim, self.initializers, posterior=False)  # .to(device)
        # NOTE: in_channels + 1 is used because the encoder for posterior net was not getting initialized properly.
        self.posterior = AxisAlignedConvGaussian(self.in_channels + 1, self.num_feat_maps, self.convs_per_block,
                                                 self.latent_dim, self.initializers, posterior=True)  # .to(device)
        self.fcomb = FComb(self.num_feat_maps, self.latent_dim, self.in_channels, self.num_classes, self.convs_fcomb,
                           initializers={'w': 'orthogonal', 'b': 'normal'}, use_tile=True)  # .to(device)

    def forward(self, patch, seg_mask, training=True):
        """
        Construct the prior latent space for the input patch and also pass it through the U-Net.
        If training is true, construct a posterior latent space also.
        """
        if training:
            self.posterior_latent_space = self.posterior.forward(patch, seg=seg_mask)     # conditioned upon GT
        self.prior_latent_space = self.prior.forward(patch)   # NOT conditioned upon the GT, just the input patch
        self.unet_features = self.unet.forward(patch, False)

    def sample(self, testing=False):
        """
        Sample a segmentation mask by picking a random sample and concatenating with U-Net's feature maps
        Difference b/w "rsample()" and "sample()" in PyTorch Distributions:
            rsample is used whenever the gradients of the distribution parameters w.r.t the functions of the samples
            need to be computed (i.e. it supports differentiation through the sampler). Useful for reparameterization
            trick in VAEs where backprop is possible through the mean and std parameters
        More on this: https://stackoverflow.com/questions/60533150/what-is-the-difference-between-sample-and-rsample
        and https://forum.pyro.ai/t/sample-vs-rsample/2344
        """
        if not testing:
            z_prior = self.prior_latent_space.rsample()
            self.z_prior_sample = z_prior
        else:
            z_prior = self.prior_latent_space.sample()
            self.z_prior_sample = z_prior

        return self.fcomb.forward(self.unet_features, z_prior)

    def reconstruct(self, use_posterior_mean=False, calc_posterior=False, z_posterior=None):
        """
        Reconstruct a segmentation map by sampling from the posterior latent space and combine it with U-Net
        use_posterior_mean: i.e. use posterior mean instead of just sampling z from Q
        calc_posterior: use a provided sample or sample fresh from the posterior latent space
        """
        if use_posterior_mean:
            z_posterior = self.posterior_latent_space.loc
        else:
            if calc_posterior:
                z_posterior = self.posterior_latent_space.rsample()

        return self.fcomb.forward(self.unet_features, z_posterior)

    def kl_divergence(self, analytic=True, calc_posterior=False, z_posterior=None):
        """
        Calculate the KL Divergence between the posterior and prior latent distributions i.e. KL(Q||P)
        analytic: calculate KL div. analytically or by sampling from the posterior
        calc_posterior: sample here if sampling is used to approximate KL or supply a sample
        """
        if analytic:
            kl_div = kl.kl_divergence(self.posterior_latent_space, self.prior_latent_space)
        else:
            if calc_posterior:
                z_posterior = self.posterior_latent_space.rsample()
            log_posterior_prob = self.posterior_latent_space.log_prob(z_posterior)
            log_prior_prob = self.prior_latent_space.log_prob(z_posterior)
            kl_div = log_posterior_prob - log_prior_prob

        return kl_div

    def elbo(self, seg_mask, analytic_kl=True, reconstruct_posterior_mean=False):
        """
        Calculate the Evidence Lower Bound of the likelihood P(Y|X)
        """
        z_posterior = self.posterior_latent_space.rsample()
        # use the posterior sample above to get a predicted segmentation mask
        self.reconstructed_mask = self.reconstruct(use_posterior_mean=reconstruct_posterior_mean, calc_posterior=False,
                                                   z_posterior=z_posterior)
        # print("Shape of Predicted Mask: ", self.reconstructed_mask.shape)   # shape: (4 x 1 x 128 x 128 x 128)

        # 1st half of the loss function
        # criterion = nn.BCEWithLogitsLoss(size_average=False, reduce=False, reduction='none')
        # reconstruction_loss = criterion(input=self.reconstructed_mask, target=seg_mask)
        # self.reconstruction_loss = torch.sum(reconstruction_loss)
        # self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        criterion = nn.BCEWithLogitsLoss()
        self.reconstruction_loss = criterion(input=self.reconstructed_mask, target=seg_mask)
        # self.reconstruction_loss = torch.sum(reconstruction_loss)
        # self.mean_reconstruction_loss = torch.mean(reconstruction_loss)

        # # TODO: use DiceLoss as the criterion instead? --> Uncomment below lines
        # criterion = DiceLoss(smooth=1.0)
        # self.reconstruction_loss = criterion(input=self.reconstructed_mask, target=seg_mask)

        # 2nd half of the loss function
        self.kl = torch.mean(self.kl_divergence(analytic=analytic_kl, calc_posterior=False, z_posterior=z_posterior))

        # Full loss
        final_loss = -(self.reconstruction_loss + self.beta * self.kl)

        print("\n")
        print(f'\tReconstruction Loss: %0.6f | KL Divergence: %0.4f | Final ELBO: %0.4f'
              % (self.reconstruction_loss.item(), self.kl.item(), final_loss.item()))

        return self.reconstructed_mask, final_loss


if __name__ == "__main__":
    inp = torch.randn(4, 2, 128, 128, 128)
    segm = torch.randn(4, 1, 128, 128, 128)
    in_channels = 3     # works for posterior net
    # in_channels = 2     # works for prior net
    convs_per_block, latent_dim = 3, 6
    num_feat_maps = [32, 64, 128, 192]
    initializers = {'w': 'he_normal', 'b': 'normal'}
    posterior = True

    net = AxisAlignedConvGaussian(in_channels, num_feat_maps, convs_per_block, latent_dim, initializers, posterior)
    net.forward(inp, segm)  # use for posterior net
    # net.forward(inp)    # use for prior net
