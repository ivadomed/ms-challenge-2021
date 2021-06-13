import os
from copy import deepcopy
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers import AdamW, get_linear_schedule_with_warmup
from ivadomed.losses import AdapWingLoss

from .datasets import MSSeg2Dataset
from .utils import split_dataset

parser = argparse.ArgumentParser(description='Script for training custom models for MSSeg2 Challenge 2021.')

# Arguments for model, data, and training
parser.add_argument('-id', '--model_id', default='transunet', type=str,
                    help='Model ID to-be-used for saving the .pt saved model file')
parser.add_argument('-dr', '--dataset_root', default='/home/GRAMES.POLYMTL.CA/uzmac/duke/projects/ivadomed/tmp_ms_challenge_2021_preprocessed', type=str,
                    help='Root path to the BIDS- and ivadomed-compatible dataset')

parser.add_argument('-ne', '--num_epochs', default=200, type=str,
                    help='Number of epochs for the training process')
parser.add_argument('-bs', '--batch_size', default=64, type=str,
                    help='Batch size of the training and validation processes')
parser.add_argument('-nw', '--num_workers', default=0, type=int,
                    help='Number of workers for the dataloaders')

parser.add_argument('--transformer_learning_rate', default=3e-5, type=float,
                    help='Learning rate for training the transformer')
parser.add_argument('--custom_learning_rate', default=1e-3, type=float,
                    help='Learning rate for training the custom additions to the transformer')
parser.add_argument('--weight_decay', type=float, default=0.01,
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
parser.add_argument('--local_rank', type=int, default=-1,
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
        device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        # Use NCCL for GPU training, process must have exclusive access to GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=rank,  world_size=world_size)

    # TODO: Initialize model
    model = None

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
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Load datasets
    dataset = MSSeg2Dataset(root=args.dataset_root,
                            patch_size=(128, 128, 128),
                            stride_size=(64, 64, 64),
                            center_crop_size=(320, 384, 512))

    train_dataset, val_dataset = split_dataset(dataset=dataset, val_size=0.3, seed=args.seed)

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
    custom_ids = ['head']
    grouped_model_parameters = [
        {'params': [param for name, param in model.named_parameters()
                    if not any(id_ in name for id_ in no_weight_decay_ids) and
                    not any(id_ in name for id_ in custom_ids)],
         'lr': args.transformer_learning_rate,
         'betas': args.betas,
         'weight_decay': args.weight_decay,
         'eps': args.eps},
        {'params': [param for name, param in model.named_parameters()
                    if any(id_ in name for id_ in no_weight_decay_ids) and
                    not any(id_ in name for id_ in custom_ids)],
         'lr': args.transformer_learning_rate,
         'betas': args.betas,
         'weight_decay': 0.0,
         'eps': args.eps},
        {'params': [param for name, param in model.named_parameters()
                    if any(id_ in name for id_ in custom_ids)],
         'lr': args.custom_learning_rate,
         'betas': args.betas,
         'weight_decay': 0.0,
         'eps': args.eps},
    ]

    # Optionally, set the base transformer to frozen
    for name, param in model.named_parameters():
        if not any(id_ in name for id_ in custom_ids):
            param.requires_grad = False

    optimizer = AdamW(grouped_model_parameters)
    num_training_steps = int(args.num_epochs * len(train_dataset) / args.batch_size)
    num_warmup_steps = 0  # int(num_training_steps * 0.02)
    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    # Setup loss function
    criterion = AdapWingLoss(theta=0.5, alpha=2.1, omega=14, epsilon=1)
    # TODO: The above params for the AdapWingLoss() are default given by `ivadomed`. Can
    #       we improve them for our application?

    # Training & Evaluation
    for i in tqdm(range(args.num_epochs), desc='Iterating over Epochs'):
        # -------------------------------- TRAINING ------------------------------------
        model.train()
        train_epoch_loss = 0.0

        for batch in tqdm(train_loader, desc='Iterating over Training Examples'):
            optimizer.zero_grad()

            x1, x2, y = batch
            x1, x2, y = x1.to(device), x2, y.to(device)

            y_hat = model(x1, x2)

            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            train_epoch_loss += loss.item()

        train_epoch_loss /= len(train_loader)

        # -------------------------------- EVALUATION ------------------------------------
        model.eval()
        val_epoch_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Iterating over Validation Examples'):
                x1, x2, y = batch
                x1, x2, y = x1.to(device), x2, y.to(device)

                y_hat = model(x1, x2)

                loss = criterion(y_hat, y)

                val_epoch_loss += loss.item()

        val_epoch_loss /= len(val_loader)

        torch.save(model.state_dict(), os.path.join(args.save, '%s.pt' % model_id))

        print('\n')  # Do this in order to go below the second tqdm line
        print(f'\tTrain Loss: %0.4f | Validation Loss: %0.4f' % (train_epoch_loss, val_epoch_loss))

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
    # Set random seed for reproducibility
    # NOTE: Settings seeds requires cuda.deterministic = True, which slows things down considerably
    # set_seed(seed=args.seed)

    main()
