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
from ivadomed.losses import AdapWingLoss, DiceLoss, GeneralizedDiceLoss, FocalTverskyLoss

from datasets import MSSeg1Dataset, MSSeg2Dataset
from utils import split_dataset, binary_accuracy, set_seeds
from models import TestModel, ModelConfig, TransUNet3D, ModifiedUNet3D

parser = argparse.ArgumentParser(description='Script for training custom models for MSSeg2 Challenge 2021.')

# Arguments for model, data, and training
parser.add_argument('-e', '--only_eval', default=False, action='store_true',
                    help='Only do evaluation, i.e. skip training!')
parser.add_argument('-id', '--model_id', default='transunet', type=str,
                    help='Model ID to-be-used for saving the .pt saved model file')
parser.add_argument('-m', '--model_type', choices=['transunet', 'unet', 'attentionunet'], default='transunet', type=str,
                    help='Model type to be used')
parser.add_argument('-dr', '--dataset_root', default='/home/GRAMES.POLYMTL.CA/u114716/duke/projects/ivadomed/tmp_ms_challenge_2021_preprocessed', type=str,
                    help='Root path to the BIDS- and ivadomed-compatible dataset')

parser.add_argument('-t', '--task', choices=['1', '2'], default='2', type=str,
                    help='Choose 1 for MSSeg1 (2016) challenge dataset (pretraining) and 2 for MSSeg2 (2021) challenge dataset (finetuning)')
parser.add_argument('-gt', '--gt_type', choices=['consensus', 'expert1', 'expert2', 'expert3', 'expert4', 'expert5', 'expert6', 'expert7', 'random', 'staple', 'average'], default='consensus', type=str,
                    help='The GT to use for the training process.')

parser.add_argument('-svs', '--subvolume_size', default=64, type=int,
                    help='Set the subvolume size to be used in training & validation')
parser.add_argument('-srs', '--stride_size', default=32, type=int,
                    help='Set the stride / translation size to be used in training & validation')
parser.add_argument('-ps', '--patch_size', default=16, type=int,
                    help='Set the patch size to be used in training & validation. (TransUNet3D-specific)')
parser.add_argument('-bnf', '--base_n_filter', default=8, type=int, help="Number of Base Filters")

parser.add_argument('-fd', '--fraction_data', default=1.0, type=float,
                    help='Fraction of data to use for the experiment. Helps with debugging.')
parser.add_argument('-fho', '--fraction_hold_out', default=0.2, type=float,
                    help='Fraction of data to hold-out of for the test phase')
parser.add_argument('-ft', '--fraction_train', default=0.8, type=float,
                    help='Fraction of data (out of kept, i.e. after hold-out) for the train phase.')
parser.add_argument('-v', '--visualize_test_preds', default=False, action='store_true',
                    help='Enable to save subvolume predictions during the test phase for visual assessment')

parser.add_argument('-ne', '--num_epochs', default=100, type=int,
                    help='Number of epochs for the training process')
parser.add_argument('-bs', '--batch_size', default=20, type=int,
                    help='Batch size of the training and validation processes')
parser.add_argument('-nw', '--num_workers', default=4, type=int,
                    help='Number of workers for the dataloaders')

parser.add_argument('-sl', '--seg_loss', choices=['dice', 'generalized_dice', 'adap_wing', 'focal_tversky_loss'], default='dice', type=str,
                    help='Select the primary loss for the segmentation task')
parser.add_argument('-bal', '--balance_strategy', choices=['none', 'naive_duplication', 'cheap_duplication', 'naive_removal'], default='naive_duplication', type=str,
                    help='The balancing strategy to employ for the training subset')

parser.add_argument('-lr', '--learning_rate', default=3e-5, type=float,
                    help='Learning rate for training the model')
parser.add_argument('-wd', '--weight_decay', type=float, default=0.01,
                    help='Weight decay (i.e. regularization) value in AdamW')
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999),
                    help='Decay terms for the AdamW optimizer')
parser.add_argument('--eps', type=float, default=1e-8,
                    help='Epsilon value for the AdamW optimizer')

parser.add_argument('-mcd', '--mc_dropout', default=False, action='store_true',
                    help='To use Monte Carlo samples for validation and testing')
parser.add_argument('-n_mc', '--num_mc_samples', default=0, type=int, help="Number of MC samples to use")

parser.add_argument('-s', '--save', default='./saved_models', type=str,
                    help='Path to the saved models directory')
parser.add_argument('-c', '--continue_from_checkpoint', default=False, action='store_true',
                    help='Load model from checkpoint and continue training')
parser.add_argument('-clp', '--continue_load_path', default=None, type=str,
                    help='Path to the trained .pt saved model file which we want to finetune / continue training on')
parser.add_argument('-ls', '--load_strategy', choices=['only_encoder', 'all'], default='all',
                    help='How to load the weights for a finetuning experiment or continuation / evaluation')
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


# Explicitly enabling (only) dropout - will be used during validation and test phases (for MC Dropout)
def enable_dropout(m):
    # # Method 1:
    # if type(m) == nn.Dropout:
    #     m.train()
    # Method 2:
    for module in m.modules():
        if module.__class__.__name__.startswith('Dropout'):
            module.train()


def main_worker(rank, world_size):
    # Configure model ID & append important / distinguishing args to model_id
    if args.mc_dropout:
        model_id = '%s-t=%s-gt=%s-bs=%d-sl=%s-bal=%s-lr=%s-svs=%d-srs=%d-bnf=%d-se=%d-nmc=%d' % \
            (args.model_id, args.task, args.gt_type, args.batch_size, args.seg_loss, args.balance_strategy,
             str(args.learning_rate), args.subvolume_size, args.stride_size, args.base_n_filter, args.seed,
             args.num_mc_samples)
    else:
        model_id = '%s-t=%s-gt=%s-bs=%d-sl=%s-bal=%s-lr=%s-svs=%d-srs=%d-bnf=%d-se=%d' % \
            (args.model_id, args.task, args.gt_type, args.batch_size, args.seg_loss, args.balance_strategy,
             str(args.learning_rate), args.subvolume_size, args.stride_size, args.base_n_filter, args.seed)
    print('MODEL ID: %s' % model_id)
    print('RANK: ', rank)

    # Configure saved models directory
    if not os.path.isdir(args.save):
        os.makedirs(args.save)

    if not args.only_eval:
        print('Trained model will be saved to: %s' % os.path.join(args.save, '%s.pt' % model_id))
    else:
        print('Running Evaluation: (1) Loss metrics on validation set, and (2) ANIMA metrics on test set')
        args.continue_from_checkpoint = True

    if args.continue_load_path is not None and not args.continue_from_checkpoint:
        raise ValueError('You need -c when you specify -clp!')

    if args.local_rank == -1:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        torch.cuda.set_device(rank)
        device = torch.device('cuda', rank)
        # Use NCCL for GPU training, process must have exclusive access to GPUs
        torch.distributed.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)

    # model = TestModel()
    cfg = ModelConfig(task=args.task,
                      subvolume_size=args.subvolume_size,
                      patch_size=args.patch_size,
                      hidden_size=512,
                      mlp_dim=512,
                      num_layers=16,
                      num_heads=16,
                      attention_dropout_rate=0.25,
                      dropout_rate=0.5,
                      layer_norm_eps=1e-7,
                      aux_clf_task=False,
                      base_n_filter=args.base_n_filter,     # TODO: Try to increase base_n_filter w/o breaking code!
                      attention_gates=True if args.model_type == 'attentionunet' else False,
                      device=device)

    model = None
    if args.model_type == 'transunet':
        model = TransUNet3D(cfg=cfg)
    elif args.model_type in ['unet', 'attentionunet']:
        model = ModifiedUNet3D(cfg=cfg)

    # Load saved model if applicable
    if args.continue_from_checkpoint:
        if not args.continue_load_path:
            load_path = os.path.join('saved_models', '%s.pt' % model_id)
        else:
            load_path = args.continue_load_path

        print('Loading learned weights from %s' % load_path)
        state_dict = torch.load(load_path)
        state_dict_ = deepcopy(state_dict)

        for param in state_dict:
            if args.load_strategy == 'only_encoder':
                # We will be initializing the decoder from scratch, so let's skip those params
                # NOTE: i.e., we will only be loading learned weights for the encoder part. This
                #       is how we see folks utilizing weights in the literature for MS segmentation.
                if 'unet_decoder' in param:
                    state_dict_.pop(param)
                    continue
            # If weights don't match bw. pretrained model and new model, we'll have to rectify it
            if not model.state_dict()[param.replace('module.', '')].shape == state_dict[param].shape:
                print('WARNING: Weight shapes DONT match for param: ', param, '. Taking care of ' +
                      'this by simply repeating weights along CHANNEL dimension! Stop the script ' +
                      'now if this is not the intended effect.')

                # In the context of Modified3DUNet, we know that the issue arises because MSSeg2016
                # dataset has one FLAIR input and hence the model has `in_channel=1`, whereas
                # MSSeg2021 dataset has two FLAIR inputs (i.e. two TPs) and hence the model has
                # `in_channel=2`. A similar problem is described here:
                # https://discuss.pytorch.org/t/how-to-change-no-of-input-channels-to-a-pretrained-model/19379/2
                # In this case, we will be repeating the weights of this specific parameter along
                # the channel dimension and reshaping it to (F, C=2, 3, 3, 3) from
                # (F, C=1, 3, 3, 3). What this means is: treat the two channels similarly to begin
                # with. This only makes sense as both TPs are FLAIR inputs and there is no need to
                # randomly initialize the other when we have learned features for one of them!
                state_dict_[param] = state_dict_[param].repeat(1, 2, 1, 1, 1)

            # Rename parameters to exclude the starting 'module.' string so they match
            # NOTE: We need this because of DataParallel saving parameters starting with 'module.'
            state_dict_[param.replace('module.', '')] = state_dict_.pop(param)

        # Load all of the available parameters
        model.load_state_dict(state_dict_, strict=False)
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
    dataset = None
    if args.task == '1':
        # Any-lesion segmentation for MSSeg2016 Challenge
        dataset = MSSeg1Dataset(root=args.dataset_root,
                                fraction_data=args.fraction_data,
                                resolution=(0.5, 0.5, 0.5),
                                center_crop_size=(384, 512, 512),
                                subvolume_size=(cfg.subvolume_size, cfg.subvolume_size, cfg.subvolume_size),
                                stride_size=(args.stride_size, args.stride_size, args.stride_size),
                                patch_size=(cfg.patch_size, cfg.patch_size, cfg.patch_size),
                                gt_type=args.gt_type,
                                use_patches=True if args.model_type == 'transunet' else False)

    elif args.task == '2':
        # New lesion segmentation for MSSeg2021 Challenge
        dataset = MSSeg2Dataset(root=args.dataset_root,
                                fraction_data=args.fraction_data,
                                fraction_hold_out=args.fraction_hold_out,
                                center_crop_size=(320, 384, 512),
                                subvolume_size=(cfg.subvolume_size, cfg.subvolume_size, cfg.subvolume_size),
                                stride_size=(args.stride_size, args.stride_size, args.stride_size),
                                patch_size=(cfg.patch_size, cfg.patch_size, cfg.patch_size),
                                gt_type=args.gt_type,
                                use_patches=True if args.model_type == 'transunet' else False,
                                results_dir='%s_RESULTS' % model_id,
                                visualize_test_preds=args.visualize_test_preds,
                                seed=args.seed)

    # NOTE: `resolution` and `center_crop_size` parameters should remain UNCHANGED for both
    #        datasets as these depend on the preprocessing step as seen in `preprocessing/`

    # Data split for train and validation phases
    train_dataset, val_dataset = split_dataset(dataset=dataset,
                                               val_size=1.0 - args.fraction_train,
                                               balance_strategy=args.balance_strategy,
                                               seed=args.seed)

    # Setup data loaders and distributed samplers depending on the training mode
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

    # Setup optimizer and scheduler
    no_weight_decay_ids = ['bias', 'LayerNorm.weight']
    grouped_model_parameters = [
        {'params': [param for name, param in model.named_parameters()
                    if not any(id_ in name for id_ in no_weight_decay_ids)],
         'lr': args.learning_rate,
         'betas': args.betas,
         'weight_decay': args.weight_decay,
         'eps': args.eps},
        {'params': [param for name, param in model.named_parameters()
                    if any(id_ in name for id_ in no_weight_decay_ids)],
         'lr': args.learning_rate,
         'betas': args.betas,
         'weight_decay': 0.0,
         'eps': args.eps},
    ]

    optimizer = AdamW(grouped_model_parameters)
    num_training_steps = int(args.num_epochs * len(train_dataset) / args.batch_size)
    num_warmup_steps = 0  # int(num_training_steps * 0.02)

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    # Setup loss function(s)
    clf_criterion = nn.CrossEntropyLoss()
    # clf_criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 100.0]).cuda())
    # NOTE: Using weighted loss to add more prevalence to positive patches

    seg_criterion = None
    if args.seg_loss == 'dice':
        seg_criterion = DiceLoss(smooth=1.0)
    elif args.seg_loss == 'generalized_dice':
        seg_criterion = GeneralizedDiceLoss(epsilon=1e-5, include_background=False)
        # TODO: `include_background = True` gives error, have to investigate!
    elif args.seg_loss == 'adap_wing':
        seg_criterion = AdapWingLoss(theta=0.5, alpha=2.1, omega=14, epsilon=1)
        # TODO: The params are default given by `ivadomed`. Can we improve them?
    elif args.seg_loss == 'focal_tversky':
        seg_criterion = FocalTverskyLoss(alpha=0.75, beta=0.25, gamma=1.5, smooth=1.0)

    # Setup other metrics for model selection
    dice_metric = DiceLoss(smooth=1.0)

    # Set scaler for mixed-precision training
    # NOTE: Check https://spell.ml/blog/mixed-precision-training-with-pytorch-Xuk7YBEAACAASJam
    scaler = torch.cuda.amp.GradScaler()

    # Train & Validate & Test Phases
    best_val_loss = float('inf')
    for i in tqdm(range(args.num_epochs), desc='Iterating over Epochs'):
        if not args.only_eval:
            # -------------------------------- TRAIN PHASE ------------------------------------
            model.train()
            dataset.train = True
            train_dataset.prepare_for_new_epoch()

            train_epoch_losses, train_epoch_accuracies = {'seg': 0.0, 'soft_dice': 0.0, 'hard_dice': 0.0}, {}
            if cfg.aux_clf_task:
                train_epoch_losses['clf'] = 0.0
                train_epoch_accuracies = {'acc': 0.0, 'pos_acc': 0.0, 'neg_acc': 0.0}

            for batch in tqdm(train_loader, desc='Iterating over Training Examples'):
                optimizer.zero_grad()

                dataset_indices, x1, x2, seg_y, clf_y = batch

                x1, x2, seg_y, clf_y = x1.to(device), x2.to(device), seg_y.to(device), clf_y.to(device)

                # Unsqueeze input patches in the channel dimension
                if dataset.use_patches:
                    x1, x2 = x1.unsqueeze(2), x2.unsqueeze(2)
                else:
                    x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)

                with torch.cuda.amp.autocast():
                    if cfg.aux_clf_task:
                        clf_y_hat, seg_y_hat = model(x1, x2)
                    else:
                        seg_y_hat = model(x1, x2)

                    seg_loss = seg_criterion(seg_y_hat, seg_y)
                    loss = seg_loss

                    if cfg.aux_clf_task:
                        clf_loss = clf_criterion(clf_y_hat.permute(0, 2, 1), clf_y)
                        loss += clf_loss

                # loss.backward()
                scaler.scale(loss).backward()

                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()

                # Update the learning rate
                scheduler.step()

                # Update metrics
                train_epoch_losses['seg'] += seg_loss.item()
                train_epoch_losses['soft_dice'] += dice_metric(seg_y_hat.detach(), seg_y.detach()).detach().item()
                train_epoch_losses['hard_dice'] += dice_metric((seg_y_hat.detach() > 0.5).float(), (seg_y.detach() > 0.5).float()).detach().item()
                if cfg.aux_clf_task:
                    train_epoch_losses['clf'] += clf_loss.item()
                    acc, pos_acc, neg_acc = binary_accuracy(clf_y_hat.detach(), clf_y.detach())
                    train_epoch_accuracies['acc'] += acc.item()
                    train_epoch_accuracies['pos_acc'] += pos_acc.item()
                    train_epoch_accuracies['neg_acc'] += neg_acc.item()

            for key in train_epoch_losses:
                train_epoch_losses[key] /= len(train_loader)
            if cfg.aux_clf_task:
                for key in train_epoch_accuracies:
                    train_epoch_accuracies[key] /= len(train_loader)

        # -------------------------------- VALIDATION PHASE ------------------------------------
        model.eval()
        dataset.train = False
        if cfg.aux_clf_task:
            val_epoch_losses['clf'] = 0.0
            val_epoch_accuracies = {'acc': 0.0, 'pos_acc': 0.0, 'neg_acc': 0.0}

        with torch.no_grad():
            if args.mc_dropout:
                print(); print("Averaging over MC samples!")
                val_epoch_losses_mc = {'mc_seg': 0.0, 'mc_soft_dice': 0.0, 'mc_hard_dice': 0.0}

                # enable only dropout layers if true
                enable_dropout(model)

                # run validation loop till the number of MC samples
                for i_mc in range(args.num_mc_samples):
                    val_epoch_losses = {'seg': 0.0, 'soft_dice': 0.0, 'hard_dice': 0.0}

                    for batch in tqdm(val_loader, desc='Iterating over Validation Examples'):
                        dataset_indices, x1, x2, seg_y, clf_y = batch
                        x1, x2, seg_y, clf_y = x1.to(device), x2.to(device), seg_y.to(device), clf_y.to(device)

                        # Unsqueeze input patches in the channel dimension
                        if dataset.use_patches:
                            x1, x2 = x1.unsqueeze(2), x2.unsqueeze(2)
                        else:
                            x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)

                        with torch.cuda.amp.autocast():
                            if cfg.aux_clf_task:
                                clf_y_hat, seg_y_hat = model(x1, x2)
                            else:
                                seg_y_hat = model(x1, x2)

                            seg_loss = seg_criterion(seg_y_hat, seg_y)
                            if cfg.aux_clf_task:
                                clf_loss = clf_criterion(clf_y_hat.permute(0, 2, 1), clf_y)

                        # Update metrics
                        val_epoch_losses['seg'] += seg_loss.item()
                        val_epoch_losses['soft_dice'] += dice_metric(seg_y_hat.detach(), seg_y.detach()).detach().item()
                        val_epoch_losses['hard_dice'] += dice_metric((seg_y_hat.detach() > 0.5).float(), (seg_y.detach() > 0.5).float()).detach().item()

                    for key in val_epoch_losses:
                        val_epoch_losses[key] /= len(val_loader)

                    # Update Monte-Carlo metrics
                    val_epoch_losses_mc['mc_seg'] += val_epoch_losses['seg']
                    val_epoch_losses_mc['mc_soft_dice'] += val_epoch_losses['soft_dice']
                    val_epoch_losses_mc['mc_hard_dice'] += val_epoch_losses['hard_dice']

                for key in val_epoch_losses_mc:
                    val_epoch_losses_mc[key] /= args.num_mc_samples

            else:
                print(); print("Standard Validation, No Monte Carlo Averaging!")
                val_epoch_losses, val_epoch_accuracies = {'seg': 0.0, 'soft_dice': 0.0, 'hard_dice': 0.0}, {}
                for batch in tqdm(val_loader, desc='Iterating over Validation Examples'):
                    dataset_indices, x1, x2, seg_y, clf_y = batch
                    x1, x2, seg_y, clf_y = x1.to(device), x2.to(device), seg_y.to(device), clf_y.to(device)

                    # Unsqueeze input patches in the channel dimension
                    if dataset.use_patches:
                        x1, x2 = x1.unsqueeze(2), x2.unsqueeze(2)
                    else:
                        x1, x2 = x1.unsqueeze(1), x2.unsqueeze(1)

                    with torch.cuda.amp.autocast():
                        if cfg.aux_clf_task:
                            clf_y_hat, seg_y_hat = model(x1, x2)
                        else:
                            seg_y_hat = model(x1, x2)

                        seg_loss = seg_criterion(seg_y_hat, seg_y)
                        if cfg.aux_clf_task:
                            clf_loss = clf_criterion(clf_y_hat.permute(0, 2, 1), clf_y)

                    # Update metrics
                    val_epoch_losses['seg'] += seg_loss.item()
                    val_epoch_losses['soft_dice'] += dice_metric(seg_y_hat.detach(), seg_y.detach()).detach().item()
                    val_epoch_losses['hard_dice'] += dice_metric((seg_y_hat.detach() > 0.5).float(), (seg_y.detach() > 0.5).float()).detach().item()
                    if cfg.aux_clf_task:
                        val_epoch_losses['clf'] += clf_loss.item()
                        acc, pos_acc, neg_acc = binary_accuracy(clf_y_hat, clf_y)
                        val_epoch_accuracies['acc'] += acc.item()
                        val_epoch_accuracies['pos_acc'] += pos_acc.item()
                        val_epoch_accuracies['neg_acc'] += neg_acc.item()

                for key in val_epoch_losses:
                    val_epoch_losses[key] /= len(val_loader)
                if cfg.aux_clf_task:
                    for key in val_epoch_accuracies:
                        val_epoch_accuracies[key] /= len(val_loader)

        print('\n')  # Do this in order to go below the second tqdm line
        if not args.only_eval:
            # Save the model file only if it is best val. loss so far
            if val_epoch_losses['seg'] < best_val_loss:
                best_val_loss = val_epoch_losses['seg']
                print('Saving best model (w.r.t. validation segmentation loss)...')
                torch.save(model.state_dict(), os.path.join(args.save, '%s.pt' % model_id))

            if cfg.aux_clf_task:
                print(f'\t[CLF] -> Train Loss: %0.4f | Validation Loss: %0.4f' % (train_epoch_losses['clf'], val_epoch_losses['clf']))

            if not args.mc_dropout:
                print(f'\t[SEG] -> Train Loss: %0.4f | Validation Loss: %0.4f' % (train_epoch_losses['seg'], val_epoch_losses['seg']))
                print(f'\t[SOFT DICE] -> Train Loss: %0.4f | Validation Loss: %0.4f' % (train_epoch_losses['soft_dice'], val_epoch_losses['soft_dice']))
                print(f'\t[HARD DICE] -> Train Loss: %0.4f | Validation Loss: %0.4f' % (train_epoch_losses['hard_dice'], val_epoch_losses['hard_dice']))
            else:
                print(f'\t[MC SEG] -> Train Loss: %0.4f | Validation Loss: %0.4f' % (train_epoch_losses['seg'], val_epoch_losses_mc['mc_seg']))
                print(f'\t[MC SOFT DICE] -> Train Loss: %0.4f | Validation Loss: %0.4f' % (train_epoch_losses['soft_dice'], val_epoch_losses_mc['mc_soft_dice']))
                print(f'\t[MC HARD DICE] -> Train Loss: %0.4f | Validation Loss: %0.4f' % (train_epoch_losses['hard_dice'], val_epoch_losses_mc['mc_hard_dice']))

        else:
            if cfg.aux_clf_task:
                print(f'\t[CLF] -> Validation Loss: %0.4f' % val_epoch_losses['clf'])
            if not args.mc_dropout:
                print(f'\t[SEG] -> Validation Loss: %0.4f' % val_epoch_losses['seg'])
                print(f'\t[SOFT DICE] -> Validation Loss: %0.4f' % val_epoch_losses['soft_dice'])
                print(f'\t[HARD DICE] -> Validation Loss: %0.4f' % val_epoch_losses['hard_dice'])
            else:
                print(f'\t[MC SEG] -> Validation Loss: %0.4f' % val_epoch_losses_mc['mc_seg'])
                print(f'\t[MC SOFT DICE] -> Validation Loss: %0.4f' % val_epoch_losses_mc['mc_soft_dice'])
                print(f'\t[MC HARD DICE] -> Validation Loss: %0.4f' % val_epoch_losses_mc['mc_hard_dice'])

        if cfg.aux_clf_task and not args.only_eval:
            print(f'\t[CLF] -> Train Accuracy: {train_epoch_accuracies["acc"] * 100:.2f}% | Validation Accuracy: {val_epoch_accuracies["acc"] * 100:.2f}%')
            print(f'\t[CLF] -> Train Pos. Accuracy: {train_epoch_accuracies["pos_acc"] * 100:.2f}% | Validation Pos. Accuracy: {val_epoch_accuracies["pos_acc"] * 100:.2f}%')
            print(f'\t[CLF] -> Train Neg. Accuracy: {train_epoch_accuracies["neg_acc"] * 100:.2f}% | Validation Neg. Accuracy: {val_epoch_accuracies["neg_acc"] * 100:.2f}%')

        # -------------------------------- TEST PHASE ------------------------------------
        # NOTE: Let's only perform the test phase when `only_eval` is specified!
        if args.only_eval:
            dataset.train = False
            if args.mc_dropout:
                enable_dropout(model)   # enabling only dropout for the test phase
            dataset.test(model=model, device=device, num_mc_samples=args.num_mc_samples)
            exit(0)

    # Cleanup DDP if applicable
    if args.local_rank != -1:
        torch.distributed.destroy_process_group()


def main():
    # Set seeds for reproducibility
    set_seeds(seed=args.seed)

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
