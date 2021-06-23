import os
import argparse
from tqdm import tqdm
from copy import deepcopy

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed

from ivadomed.losses import AdapWingLoss, DiceLoss

from datasets import MSSeg2Dataset
from probabilistic_unet import ProbabilisticUnet
from utils import l2_regularization, split_dataset

parser = argparse.ArgumentParser(description='Script for training Probabilistic U-Net for MSSeg2 Challenge 2021.')

# Arguments for model, data, and training
parser.add_argument('-id', '--model_id', default='punet', type=str,
                    help='Model ID to-be-used for saving the .pt saved model file')
parser.add_argument('-dr', '--dataset_root',
                    default='/home/GRAMES.POLYMTL.CA/u114716/duke/projects/ivadomed/tmp_ms_challenge_2021_preprocessed',
                    type=str, help='Root path to the BIDS- and ivadomed-compatible dataset')
parser.add_argument('-fd', '--fraction_data', default=1.0, type=float, help='Fraction of data to use (for debugging)')

parser.add_argument('-ne', '--num_epochs', default=20, type=int, help='Number of epochs for the training process')
parser.add_argument('-bs', '--batch_size', default=8, type=int, help='Batch size for training and validation processes')
parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of workers for the dataloaders')

parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='Learning rate for training')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999), help='Decay terms for the AdamW optimizer')
parser.add_argument('-wd', '--weight_decay', type=float, default=0.01, help='Decay value in AdamW')
parser.add_argument('--eps', type=float, default=1e-8, help='Epsilon value for the AdamW optimizer')

# PU-Net-specific Arguments
parser.add_argument('-ldim', '--latent_dim', default=6, type=int, help='Dimensionality of the latent space')
parser.add_argument('-nfc', '--num_fcomb', default=4, type=int, help='No. of 1x1 conv blocks to concat with UNet')
# TODO: change beta values and see how the training progresses!
parser.add_argument('--beta', default=10.0, type=float, help='Weighting factor for the ELBO loss function')

parser.add_argument('-sv', '--save', default='../results', type=str, help='Path to the saved models directory')
parser.add_argument('-c', '--continue_from_checkpoint', default=False, action='store_true',
                    help='Load model from checkpoint and continue training')
parser.add_argument('-se', '--seed', default=42, type=int, help='Set seeds for reproducibility')

# Arguments for parallelization
parser.add_argument('-loc', '--local_rank', type=int, default=-1,
                    help='Local rank for distributed training on GPUs; set != -1 to start distributed GPU training')
parser.add_argument('--master_addr', type=str, default='localhost',
                    help='Address of master; master must be able to accept network traffic on the address and port')
parser.add_argument('--master_port', type=str, default='29500', help='Port that master is listening on')

args = parser.parse_args()


def main_worker(rank, world_size):
    # Configure model ID
    model_id = args.model_id
    print('MODEL ID: %s' % model_id)
    print('RANK: ', rank)

    # Configure saved models directory
    if not os.path.isdir(args.save):
        os.makedirs(args.save)
    print('Trained model will be saved to: %s' % args.save)

    if args.local_rank == -1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        # Use NCCL for GPU training, process must have exclusive access to GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

    model = ProbabilisticUnet(in_channels=2, num_classes=1, num_feat_maps=[32, 64, 128, 192],
                              latent_dim=args.latent_dim, convs_fcomb=args.num_fcomb, beta=args.beta)

    # Load saved model if applicable
    if args.continue_from_checkpoint:
        load_path = os.path.join('saved_models', '%s.pt' % model_id)
        print('Loading learned weights from %s' % load_path)
        state_dict = torch.load(load_path)
        state_dict_ = deepcopy(state_dict)
        # Rename parameters to exclude the starting 'module.' string so they match
        # NOTE: We have to do this because of DataParallel saving parameters starting with 'module.'
        for param in state_dict:
            state_dict_[param.replace('module.', '')] = state_dict_.pop(param)
        model.load_state_dict(state_dict_)
    else:
        print('Initializing model from scratch')

    if torch.cuda.device_count() > 1 and args.local_rank == -1:
        model = nn.DataParallel(model)
        model.to(device)
    elif args.local_rank == -1:
        model.to(device)
    else:
        model.to(device)
        try:
            from apex.parallel import DistributedDataParallel as DDP
            print("Found Apex!")
            model = DDP(model)
        except ImportError:
            from torch.nn.parallel import DistributedDataParallel as DDP
            print("Using PyTorch DDP - could not find Apex")
            model = DDP(model, device_ids=[rank])

    # Load datasets
    dataset = MSSeg2Dataset(root=args.dataset_root, subvolume_size=(128, 128, 128), stride_size=(64, 64, 64),
                            center_crop_size=(320, 384, 512), fraction_data=args.fraction_data, num_gt_experts=4)

    train_dataset, val_dataset = split_dataset(dataset=dataset, val_size=0.3)
    # TODO: We also need test set, right?

    if args.local_rank == -1:
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  pin_memory=True, num_workers=args.num_workers)
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=False,
                                pin_memory=True, num_workers=args.num_workers)
    else:
        train_sampler = DistributedSampler(dataset=train_dataset, num_replicas=world_size, rank=rank)
        val_sampler = DistributedSampler(dataset=val_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size,
                                  shuffle=False, pin_memory=True,
                                  num_workers=args.num_workers, sampler=train_sampler)
        # NOTE: Train loader's shuffle is made False; using DistributedSampler instead
        val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch_size,
                                shuffle=False, pin_memory=True,
                                num_workers=args.num_workers, sampler=val_sampler)

    # # Setup optimizer
    # no_weight_decay_ids = ['bias', 'LayerNorm.weight']
    # grouped_model_parameters = [
    #     {'params': [param for name, param in model.named_parameters()
    #                 if not any(id_ in name for id_ in no_weight_decay_ids)],
    #      'lr': args.transformer_learning_rate,
    #      'betas': args.betas,
    #      'weight_decay': args.weight_decay,
    #      'eps': args.eps},
    #     {'params': [param for name, param in model.named_parameters()
    #                 if any(id_ in name for id_ in no_weight_decay_ids)],
    #      'lr': args.transformer_learning_rate,
    #      'betas': args.betas,
    #      'weight_decay': 0.0,
    #      'eps': args.eps},
    # ]
    #
    # optimizer = AdamW(grouped_model_parameters)
    # num_training_steps = int(args.num_epochs * len(train_dataset) / args.batch_size)
    # num_warmup_steps = 0  # int(num_training_steps * 0.02)
    # scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
    #                                             num_warmup_steps=num_warmup_steps,
    #                                             num_training_steps=num_training_steps)
    #
    # # Setup loss function
    # criterion = AdapWingLoss(theta=0.5, alpha=2.1, omega=14, epsilon=1)

    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=0.0)
    # TODO: use a learning rate scheduler to decay lr upto 1e-6

    # Setup other metrics
    dice_metric = DiceLoss(smooth=1.0)

    # for mixed-precision training
    scaler = torch.cuda.amp.GradScaler()

    # Training & Evaluation
    for i in tqdm(range(args.num_epochs), desc='Iterating over Epochs'):
        # -------------------------------- TRAINING ------------------------------------
        model.train()
        train_epoch_loss, train_epoch_dice = 0.0, 0.0

        for batch in tqdm(train_loader, desc='Iterating over Training Examples'):
            # Glossary --> B: batch_size; SV: subvolume_size; P: patch_size
            x1, x2, y = batch
            # print(x1.shape, "\t", x2.shape, "\t", y.shape)
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)  # size of x1,x2, y: (B x P x P x P)

            # Unsqueeze input patches in the channel dimension
            x1, x2, y = x1.unsqueeze(dim=1), x2.unsqueeze(dim=1), torch.unsqueeze(y, dim=1)   # size:(B x 1 x P x P x P)
            # print(x1.shape, "\t", x2.shape, "\t", y.shape)

            # Concatenate time-points to get single input which is "2 x " the original input size
            x = torch.cat([x1, x2], dim=1).to(device)       # size of x: (B x 2 x P x P x P)
            # print(x.shape)

            with torch.cuda.amp.autocast():
                model.forward(patch=x, seg_mask=y, training=True)

                y_hat, elbo = model.elbo(y)     # y_hat is the reconstructed mask - shape: (B x 1 x P x P x P)
                reg_loss = l2_regularization(model.posterior) + l2_regularization(model.prior) + \
                    l2_regularization(model.fcomb.layers)
                loss = -elbo + 1e-5 * reg_loss

            optimizer.zero_grad()
            # Backprop in Mixed-Precision training
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # Backprop in Standard training
            # loss.backward()
            # optimizer.step()

            train_epoch_loss += loss.item()
            train_epoch_dice += dice_metric(y_hat, y).item()

        train_epoch_loss /= len(train_loader)
        train_epoch_dice /= len(train_loader)

        # -------------------------------- EVALUATION ------------------------------------
        # Evaluation done by sampling from the prior latent distribution "NP" no. of times combining with
        # U-Net's features. These NP predictions are concatenated and their mean is taken, which is the final predicted
        # segmentation. The Dice score is calculated b/w this prediction and the GT.
        # TODO: A random GT out of the 4 is chosen now. Should it be compared with the consensus (i.e. the 5th) GT?

        model.eval()
        # val_epoch_loss = 0.0      # not calculating loss in validation, only Dice
        val_epoch_dice = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Iterating over Validation Examples'):
                x1, x2, y = batch
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)

                x1, x2 = x1.unsqueeze(dim=1), x2.unsqueeze(dim=1)
                x = torch.cat([x1, x2], dim=1).to(device)

                with torch.cuda.amp.autocast():
                    model.forward(patch=x, seg_mask=None, training=False)

                    num_predictions = 5     # NP
                    predictions = []
                    for _ in range(num_predictions):
                        mask_pred = model.sample(testing=True)
                        # TODO: this line below gets hard predictions. Use just sigmoid and see how Dice performs.
                        mask_pred = torch.sigmoid(mask_pred)    # getting a soft pred. ; shape: (B x 1 x P x P x P)
                        # uncomment the line below for getting hard predictions.
                        # mask_pred = (torch.sigmoid(mask_pred) > 0.5).float()    # shape: (B x 1 x P x P x P)
                        predictions.append(mask_pred)
                    predictions = torch.cat(predictions, dim=1)     # shape: (B x NP x P x P x P)

                    y_pred = torch.squeeze(torch.mean(predictions, dim=1))    # y_pred is the mean of all NP predictions
                    # shape: (B x P x P x P)

                val_epoch_dice += dice_metric(y_pred, y)

        # val_epoch_loss /= len(val_loader)
        val_epoch_dice /= len(val_loader)

        torch.save(model.state_dict(), os.path.join(args.save, '%s.pt' % model_id))

        print('\n')  # Do this in order to go below the second tqdm line
        print(f'\tTrain Loss: %0.4f | Train Dice: %0.6f' % (train_epoch_loss, train_epoch_dice))
        print(f'\tValidation Dice: %0.6f' % val_epoch_dice)

        # # Apply learning rate decay before the beginning of next epoch if applicable
        # scheduler.step()

    # Cleanup DDP if applicable
    if args.local_rank != -1:
        torch.distributed.destroy_process_group()


def main():
    # Configure number of GPUs to be used
    n_gpus = torch.cuda.device_count()
    print('We are using %d GPUs' % n_gpus)

    # Spawn the training process
    if args.local_rank == -1:
        main_worker(rank=-1, world_size=1)
    else:
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
        print('Spawning...')
        # Number of processes spawned is equal to the number of GPUs available
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus,))


if __name__ == '__main__':
    main()
