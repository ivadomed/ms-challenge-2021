import os
from copy import deepcopy
import argparse
from typing import Dict

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AdamW, get_linear_schedule_with_warmup
from ivadomed.losses import AdapWingLoss, DiceLoss

from datasets import MSSeg2Dataset
from utils import split_dataset, binary_accuracy
from models import TestModel, TransUNet3D, ModelConfig

parser = argparse.ArgumentParser(description='Script for training custom models for MSSeg2 Challenge 2021.')

# Arguments for model, data, and training
parser.add_argument('-id', '--model_id', default='transunet', type=str,
                    help='Model ID to-be-used for saving the .pt saved model file')
parser.add_argument('-dr', '--dataset_root', default='/home/GRAMES.POLYMTL.CA/uzmac/duke/projects/ivadomed/tmp_ms_challenge_2021_preprocessed', type=str,
                    help='Root path to the BIDS- and ivadomed-compatible dataset')
parser.add_argument('-fd', '--fraction_data', default=1.0, type=float,
                    help='Fraction of data to use for the experiment. Helps with debugging.')
parser.add_argument('-gt', '--gt_type', choices=['consensus', 'expert1', 'expert2', 'expert3', 'expert4'], default='consensus', type=str,
                    help='The GT to use for the training process.')

parser.add_argument('-ne', '--num_epochs', default=200, type=int,
                    help='Number of epochs for the training process')
parser.add_argument('-bs', '--batch_size', default=20, type=int,
                    help='Batch size of the training and validation processes')
parser.add_argument('-nw', '--num_workers', default=4, type=int,
                    help='Number of workers for the dataloaders')

parser.add_argument('-tlr', '--transformer_learning_rate', default=3e-5, type=float,
                    help='Learning rate for training the transformer')
parser.add_argument('-clr', '--custom_learning_rate', default=1e-3, type=float,
                    help='Learning rate for training the custom additions to the transformer')
parser.add_argument('-wd', '--weight_decay', type=float, default=0.01,
                    help='Weight decay (i.e. regularization) value in AdamW')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='Decay terms for the AdamW optimizer')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='Epsilon value for the AdamW optimizer')

parser.add_argument('-sv', '--save', default='./saved_models', type=str,
                    help='Path to the saved models directory')
parser.add_argument('-c', '--continue_from_checkpoint', default=False, action='store_true',
                    help='Load model from checkpoint and continue training')
parser.add_argument('-se', '--seed', default=42, type=int,
                    help='Set seeds for reproducibility')

# Arguments for parallelization
parser.add_argument('-loc', '--local_rank', type=int, default=-1,
                    help='Local rank for distributed training on GPUs; set to != -1 to start distributed GPU training')
parser.add_argument('--master_addr', type=str, default='localhost',
                    help='Address of master; master must be able to accept network traffic on the address and port')
parser.add_argument('--master_port', type=str, default='29500',
                    help='Port that master is listening on')

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

    # model = TestModel()
    cfg = ModelConfig(subvolume_size=128,
                      patch_size=32,
                      hidden_size=64,
                      mlp_dim=256,
                      num_layers=8,
                      num_heads=8,
                      attention_dropout_rate=0.3,
                      dropout_rate=0.3,
                      layer_norm_eps=1e-6,
                      head_channels=512,
                      device=device)
    model = TransUNet3D(cfg=cfg)

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
            model = DDP(model, device_ids=[rank], find_unused_parameters=False)
            # TODO: `find_unused_parameters` might have to be True for certain model types!

    # Load datasets
    dataset = MSSeg2Dataset(root=args.dataset_root,
                            center_crop_size=(320, 384, 512),
                            subvolume_size=(128, 128, 128),
                            stride_size=(64, 64, 64),
                            patch_size=(32, 32, 32),
                            fraction_data=args.fraction_data,
                            gt_type=args.gt_type,
                            use_patches=True,
                            seed=args.seed)

    train_dataset, val_dataset = split_dataset(dataset=dataset, val_size=0.3, seed=args.seed)
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

    # Setup optimizer
    no_weight_decay_ids = ['bias', 'LayerNorm.weight']
    grouped_model_parameters = [
        {'params': [param for name, param in model.named_parameters()
                    if not any(id_ in name for id_ in no_weight_decay_ids)],
         'lr': args.transformer_learning_rate,
         'betas': args.betas,
         'weight_decay': args.weight_decay,
         'eps': args.eps},
        {'params': [param for name, param in model.named_parameters()
                    if any(id_ in name for id_ in no_weight_decay_ids)],
         'lr': args.transformer_learning_rate,
         'betas': args.betas,
         'weight_decay': 0.0,
         'eps': args.eps},
    ]

    optimizer = AdamW(grouped_model_parameters)
    num_training_steps = int(args.num_epochs * len(train_dataset) / args.batch_size)
    num_warmup_steps = 0  # int(num_training_steps * 0.02)
    # TODO: Explore warm-up
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    # Setup loss function(s)
    clf_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0]).cuda())
    # NOTE: Using weighted loss to add more prevalence to positive patches

    seg_criterion = AdapWingLoss(theta=0.5, alpha=2.1, omega=14, epsilon=1)
    # TODO: The above params for the AdapWingLoss() are default given by `ivadomed`. Can
    #       we improve them for our application?

    # Setup other metrics
    dice_metric = DiceLoss(smooth=1.0)

    # Set scaler for mixed-precision training
    # NOTE: Check https://spell.ml/blog/mixed-precision-training-with-pytorch-Xuk7YBEAACAASJam
    scaler = torch.cuda.amp.GradScaler()

    # Training & Evaluation
    for i in tqdm(range(args.num_epochs), desc='Iterating over Epochs'):
        # -------------------------------- TRAINING ------------------------------------
        model.train()
        dataset.train = True
        train_epoch_losses = {'clf': 0.0, 'seg': 0.0, 'dice': 0.0}
        train_epoch_accuracies = {'acc': 0.0, 'pos_acc': 0.0, 'neg_acc': 0.0}

        for batch in tqdm(train_loader, desc='Iterating over Training Examples'):
            optimizer.zero_grad()

            x1, x2, seg_y, clf_y = batch
            x1, x2, seg_y, clf_y = x1.to(device), x2.to(device), seg_y.to(device), clf_y.to(device)

            # Unsqueeze input patches in the channel dimension
            x1, x2 = x1.unsqueeze(2), x2.unsqueeze(2)

            with torch.cuda.amp.autocast():
                clf_y_hat, seg_y_hat = model(x1, x2)

                # Get baseline estimate of segmentation maps with classification logits
                # clf_seg_y_hat = torch.zeros_like(seg_y).to(device)
                # clf_seg_y_hat += torch.argmax(clf_y_hat, dim=-1)[:, cfg.num_patches:]
                # loss = seg_criterion(clf_seg_y_hat, seg_y)

                clf_loss = clf_criterion(clf_y_hat.permute(0, 2, 1), clf_y)
                seg_loss = seg_criterion(seg_y_hat, seg_y)
                loss = clf_loss + seg_loss

            # loss.backward()
            scaler.scale(loss).backward()

            # optimizer.step()
            scaler.step(optimizer)
            scaler.update()

            # Update metrics
            train_epoch_losses['clf'] += clf_loss.item()
            train_epoch_losses['seg'] += seg_loss.item()
            train_epoch_losses['dice'] += dice_metric(seg_y_hat, seg_y).item()

            acc, pos_acc, neg_acc = binary_accuracy(clf_y_hat, clf_y)
            train_epoch_accuracies['acc'] += acc.item()
            train_epoch_accuracies['pos_acc'] += pos_acc.item()
            train_epoch_accuracies['neg_acc'] += neg_acc.item()

        for key in train_epoch_losses:
            train_epoch_losses[key] /= len(train_loader)
        for key in train_epoch_accuracies:
            train_epoch_accuracies[key] /= len(train_loader)
        # -------------------------------- EVALUATION ------------------------------------
        model.eval()
        dataset.train = False
        val_epoch_losses = {'clf': 0.0, 'seg': 0.0, 'dice': 0.0}
        val_epoch_accuracies = {'acc': 0.0, 'pos_acc': 0.0, 'neg_acc': 0.0}

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Iterating over Validation Examples'):
                x1, x2, seg_y, clf_y = batch
                x1, x2, seg_y, clf_y = x1.to(device), x2.to(device), seg_y.to(device), clf_y.to(device)

                # Unsqueeze input patches in the channel dimension
                x1, x2 = x1.unsqueeze(2), x2.unsqueeze(2)

                with torch.cuda.amp.autocast():
                    clf_y_hat, seg_y_hat = model(x1, x2)

                    # Get baseline estimate of segmentation maps with classification logits
                    # clf_seg_y_hat = torch.zeros_like(seg_y).float().to(device)
                    # clf_seg_y_hat += torch.argmax(clf_y_hat, dim=-1)[:, cfg.num_patches:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
                    # clf_seg_y_hat += torch.argmax(clf_y_hat, dim=-1)[:, cfg.num_patches:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
                    # clf_seg_y_hat *= torch.max(clf_y_hat, dim=-1).values[:, cfg.num_patches:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).float()
                    # print(seg_y.shape)
                    # print(torch.argmax(clf_y_hat, dim=-1)[:, cfg.num_patches:].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
                    # TODO: Do the above unsqueeze more elegantly...
                    # loss = seg_criterion(clf_seg_y_hat, seg_y)

                    clf_loss = clf_criterion(clf_y_hat.permute(0, 2, 1), clf_y)
                    seg_loss = seg_criterion(seg_y_hat, seg_y)
                    loss = clf_loss + seg_loss

                # Update metrics
                val_epoch_losses['clf'] += clf_loss.item()
                val_epoch_losses['seg'] += seg_loss.item()
                val_epoch_losses['dice'] += dice_metric(seg_y_hat, seg_y).item()

                acc, pos_acc, neg_acc = binary_accuracy(clf_y_hat, clf_y)
                val_epoch_accuracies['acc'] += acc.item()
                val_epoch_accuracies['pos_acc'] += pos_acc.item()
                val_epoch_accuracies['neg_acc'] += neg_acc.item()

        for key in val_epoch_losses:
            val_epoch_losses[key] /= len(val_loader)
        for key in val_epoch_accuracies:
            val_epoch_accuracies[key] /= len(val_loader)

        torch.save(model.state_dict(), os.path.join(args.save, '%s.pt' % model_id))

        print('\n')  # Do this in order to go below the second tqdm line
        print(f'\t[CFL] -> Train Loss: %0.4f | Validation Loss: %0.4f' % (train_epoch_losses['clf'], val_epoch_losses['clf']))
        print(f'\t[SEG] -> Train Loss: %0.4f | Validation Loss: %0.4f' % (train_epoch_losses['seg'], val_epoch_losses['seg']))
        print(f'\t[DICE] -> Train Loss: %0.4f | Validation Loss: %0.4f' % (train_epoch_losses['dice'], val_epoch_losses['dice']))

        print(f'\t[CLF] -> Train Accuracy: {train_epoch_accuracies["acc"] * 100:.2f}% | Validation Accuracy: {val_epoch_accuracies["acc"] * 100:.2f}%')
        print(f'\t[CLF] -> Train Pos. Accuracy: {train_epoch_accuracies["pos_acc"] * 100:.2f}% | Validation Pos. Accuracy: {val_epoch_accuracies["pos_acc"] * 100:.2f}%')
        print(f'\t[CLF] -> Train Neg. Accuracy: {train_epoch_accuracies["neg_acc"] * 100:.2f}% | Validation Neg. Accuracy: {val_epoch_accuracies["neg_acc"] * 100:.2f}%')

        # Apply learning rate decay before the beginning of next epoch if applicable
        scheduler.step()

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
        mp.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, ))


if __name__ == '__main__':
    main()
