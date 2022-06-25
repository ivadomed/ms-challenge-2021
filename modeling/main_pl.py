from cgi import test
import os
import shutil
import tempfile
import argparse
from datetime import datetime
import subprocess
from collections import defaultdict
import xml.etree.ElementTree as ET

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import wandb
import torch
import pytorch_lightning as pl

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.losses import DiceCELoss, DiceLoss, FocalLoss
from monai.inferers import sliding_window_inference
from monai.networks.nets import UNet, DynUNet, BasicUNet, SegResNet, UNETR
from monai.data import (DataLoader, CacheDataset, load_decathlon_datalist, decollate_batch, list_data_collate)
from monai.transforms import (AsDiscrete, AddChanneld, Compose, CropForegroundd, LoadImaged, Orientationd, RandFlipd, 
                    RandCropByPosNegLabeld, RandShiftIntensityd, ScaleIntensityRanged, Spacingd, RandRotate90d, ToTensord,
                    SpatialPadd, NormalizeIntensityd, EnsureType, RandScaleIntensityd, RandWeightedCropd, EnsureChannelFirstd,
                    AsDiscreted, RandSpatialCropSamplesd, HistogramNormalized, EnsureTyped, Invertd, SaveImaged, SaveImage)


# print_config()

# create a "model"-agnostic class with PL to use different models on both datasets
class Model(pl.LightningModule):
    def __init__(self, args, fold_num, data_root, net, loss_function, optimizer_class):
        super().__init__()
        self.args = args
        self.save_hyperparameters(ignore=['net', 'loss_function'])
        
        self.fold_num = fold_num
        self.root = data_root
        self.lr = args.learning_rate
        self.net = net
        self.loss_function = loss_function
        self.optimizer_class = optimizer_class

        self.best_val_dice, self.best_val_epoch = 0, 0
        # self.check_val = args.check_val_every_n_epochs
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []

        # define cropping and padding dimensions
        self.voxel_cropping_size = (args.patch_size,) * 3 
        # self.spatial_padding_size = (320, 384, 512)
        self.inference_roi_size = (args.patch_size,) * 3 

        # define post-processing transforms for validation
        self.val_post_pred = Compose([EnsureType(), AsDiscrete(argmax=True, to_onehot=2)])
        self.val_post_label = Compose([EnsureType(), AsDiscrete(to_onehot=2)])

        # define evaluation metric
        self.dice_metric = DiceMetric(include_background=False, reduction="mean", get_not_nans=False)

        # Get the ANIMA binaries path
        cmd = r'''grep "^anima = " ~/.anima_4.1.1/config.txt | sed "s/.* = //"'''
        self.anima_binaries_path = subprocess.check_output(cmd, shell=True).decode('utf-8').strip('\n')
        print('ANIMA Binaries Path: ', self.anima_binaries_path)

    def forward(self, x):
        return self.net(x)

    def prepare_data(self):
        split_JSON = f"dataset_fold-{self.fold_num}.json"
        datasets = self.root + split_JSON

        # load datasets (they are in Decathlon datalist format)
        datalist = load_decathlon_datalist(datasets, True, "training")
        val_files = load_decathlon_datalist(datasets, True, "validation")
        test_files = load_decathlon_datalist(datasets, True, "test")

        # define keys for applying the transforms
        imgs_keys = ["image_ses1", "image_ses2"]
        if self.args.num_gts == 1:
            # use only the consensus GT
            label_keys = ["label_c"]
        else:
            # use the GTs from the other raters + the consensus
            label_keys = ["label_1", "label_2", "label_3", "label_4", "label_c"]
        
        combined_keys = imgs_keys+label_keys
        
        # for Spacingd transformation
        if len(combined_keys) == 3: 
            resampling_mode = ("bilinear", "bilinear", "nearest")
        else:
            # for all raters + consensus
            resampling_mode = ("bilinear", "bilinear", "nearest", "nearest", "nearest", "nearest", "nearest")
    
        # define training and validation transforms
        # NOTE: Don't need to worry about SpatialPadding/CenterCropping because we're now cropping <voxel_cropping_size> sized patches
        # based on the label map as the weight and feeding to the model.
        
        if not args.only_eval:
            train_transforms = Compose([   
                LoadImaged(keys=combined_keys),
                AddChanneld(keys=combined_keys),
                Orientationd(keys=combined_keys, axcodes="RAS"),
                Spacingd(keys=combined_keys,pixdim=(0.75, 0.75, 0.75), mode=resampling_mode),
                CropForegroundd(keys=combined_keys, source_key=imgs_keys[0]),     # crops >0 values with a bounding box
                # RandCropByPosNegLabeld(
                #     keys=combined_keys, label_key="label_c", 
                #     spatial_size=self.voxel_cropping_size,
                #     pos=3, neg=1, 
                #     num_samples=4,  # if num_samples=4, then 4 samples/image are randomly generated
                #     image_key=imgs_keys[0], image_threshold=0.), 
                RandWeightedCropd(keys=combined_keys, w_key="label_c", 
                    spatial_size=self.voxel_cropping_size, num_samples=args.num_samples_per_volume),
                RandFlipd(keys=combined_keys, spatial_axis=[0], prob=0.50,),
                RandFlipd(keys=combined_keys, spatial_axis=[1], prob=0.50,),
                RandFlipd(keys=combined_keys,spatial_axis=[2],prob=0.50,),
                RandRotate90d(keys=combined_keys, prob=0.10, max_k=3,),
                # RandShiftIntensityd(keys=["image"], offsets=0.10, prob=1.0,),
                # RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
                HistogramNormalized(keys=imgs_keys, mask=None),
                NormalizeIntensityd(keys=imgs_keys, nonzero=False, channel_wise=True),
                ToTensord(keys=combined_keys), 
            ])

            val_transforms = Compose([
                LoadImaged(keys=combined_keys),
                AddChanneld(keys=combined_keys),
                Orientationd(keys=combined_keys, axcodes="RAS"),
                Spacingd(keys=combined_keys,pixdim=(0.75, 0.75, 0.75), mode=resampling_mode,),
                # ScaleIntensityRanged(keys=["image"], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
                CropForegroundd(keys=combined_keys, source_key=imgs_keys[0]),
                HistogramNormalized(keys=imgs_keys, mask=None),
                NormalizeIntensityd(keys=imgs_keys, nonzero=False, channel_wise=True),
                ToTensord(keys=combined_keys),
            ])
            self.train_ds = CacheDataset(data=datalist, transform=train_transforms, cache_num=20, num_workers=8)
            self.val_ds = CacheDataset(data=val_files, transform=val_transforms, cache_num=20, cache_rate=1.0, num_workers=4)

        else:
            # Load these only for testing
            test_transforms = Compose([
                LoadImaged(keys=combined_keys),
                AddChanneld(keys=combined_keys),
                Orientationd(keys=combined_keys, axcodes="RAS"),
                # Spacingd(keys=imgs_keys,pixdim=(0.75, 0.75, 0.75), mode=("bilinear", "bilinear"),),
                # CropForegroundd(keys=combined_keys, source_key=label_keys[0]),
                # HistogramNormalized(keys=imgs_keys, mask=None),   # should this be uncommented or not?
                NormalizeIntensityd(keys=imgs_keys, nonzero=False, channel_wise=True),
                ToTensord(keys=combined_keys),
            ])
            
            # define post-processing transforms for testing; taken (with explanations) from 
            # https://github.com/Project-MONAI/tutorials/blob/main/3d_segmentation/torch/unet_inference_dict.py#L66
            self.test_post_pred = Compose([
                EnsureTyped(keys=["pred", "label_c"]),
                Invertd(
                keys="pred",  # invert the `pred` data field, also support multiple fields
                transform=test_transforms,
                orig_keys=["image_ses2"],  
                        # get the previously applied pre_transforms information on the `image_sesX` data field,
                        # then invert `pred` based on this information. we can use same info for 
                        # multiple fields, also support different orig_keys for different fields
                meta_keys="pred_meta_dict",  # key field to save inverted meta data, every item maps to `keys`
                orig_meta_keys=["image_ses2_meta_dict"],  
                        # get the meta data from `img_meta_dict` field when inverting,
                        # for example, may need the `affine` to invert `Spacingd` transform,
                        # multiple fields can use the same meta data to invert
                meta_key_postfix="meta_dict",  # if `meta_keys=None`, use "{keys}_{meta_key_postfix}" as the meta key,
                                            # if `orig_meta_keys=None`, use "{orig_keys}_{meta_key_postfix}",
                                            # otherwise, no need this arg during inverting
                nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                                    # to ensure a smooth output, then execute `AsDiscreted` transform
                to_tensor=True, 
            ),
            AsDiscreted(keys="pred", argmax=True, threshold=0.5), #, to_onehot=2), ANIMA only needs binary predictions
            AsDiscreted(keys="label_c", threshold=0.5), # ANIMA only needs binary predictions
            SaveImaged(keys="pred", meta_keys="pred_meta_dict", 
                    output_dir=os.path.join(self.args.results_dir, self.args.model, args.best_model_name[:-5]), 
                    output_postfix="pred", resample=False),
            SaveImaged(keys="label_c", meta_keys="pred_meta_dict", 
                    output_dir=os.path.join(self.args.results_dir, self.args.model, args.best_model_name[:-5]), 
                    output_postfix="gt", resample=False),
            ])

            # define training and validation dataloaders
            self.test_ds = CacheDataset(data=test_files, transform=test_transforms, cache_num=5, cache_rate=1.0, num_workers=4)


    def train_dataloader(self):
        # NOTE: if num_samples=4 in RandCropByPosNegLabeld and batch_size=2, then 2 x 4 images are generated for network training
        return DataLoader(self.train_ds, batch_size=self.args.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                            collate_fn=list_data_collate)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=True)
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.args.T_0, eta_min=1e-5)
        return [optimizer] #, [scheduler]

    def _compute_loss(self, preds, labels):

        # if self.args.dataset == 'zurich':
        #     # convert labels to binary masks to calculate the weights for CELoss
        #     # and to convert to one-hot encoding for DiceLoss
        #     labels = (labels > 0.3).long()
        # print(f"labels binary: {labels}")

        # # define loss functions
        # if self.loss_function == 'dice':
        #     criterion = DiceLoss(to_onehot_y=True, softmax=True)
        # elif self.loss_function in ['dice_ce', 'dice_ce_sq']:                
        #     # compute weights for cross entropy loss
        #     labels_b = labels[0]
        #     normalized_ce_weights = get_ce_weights(labels_b.detach().cpu())
        #     # print(f"normed ce weights: {normalized_ce_weights}")
        #     if self.loss_function == 'dice_ce':
        #         criterion = DiceCELoss(to_onehot_y=True, softmax=True, ce_weight=normalized_ce_weights)
        #     else:
        #         criterion = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, ce_weight=normalized_ce_weights)
        # loss = criterion(preds, labels)
        
        loss = self.loss_function(preds, labels)

        return loss

    def training_step(self, batch, batch_idx):
        if self.args.num_gts == 1:
            input1, input2, labels = (batch["image_ses1"], batch["image_ses2"], batch["label_c"])
            inputs = torch.cat([input1, input2], dim=1)
        # Not considering the case of 5 GTs because the consensus GT is already combines the other 4
        
        output = self.forward(inputs)

        # calculate training loss
        loss = self._compute_loss(output, labels)
        
        return {
            "loss": loss,
            "train_number": len(inputs)
        }

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log('train_loss', avg_loss, on_step=False, on_epoch=True)

        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    
    def validation_step(self, batch, batch_idx):
        if self.args.num_gts == 1:
            input1, input2, labels = (batch["image_ses1"], batch["image_ses2"], batch["label_c"])
            images = torch.cat([input1, input2], dim=1)

        inference_roi_size = self.inference_roi_size
        sw_batch_size = 4
        # NOTE: shape of outputs is the actual shape of the original subject (which is cool as we 
        # now don't need to worry about creating patches and then mapping them back!)
        outputs = sliding_window_inference(images, inference_roi_size, sw_batch_size, self.forward, overlap=0.75,) 
        # outputs shape: (B, C, <original H x W x D>)
        
        # calculate validation loss
        loss = self._compute_loss(outputs, labels)
        
        # post-process for calculating the evaluation metric
        post_outputs = [self.val_post_pred(i) for i in decollate_batch(outputs)]
        post_labels = [self.val_post_label(i) for i in decollate_batch(labels)]
        # post_outputs shape = post_labels shape = (C, <original H x W x D>)
        self.dice_metric(y_pred=post_outputs, y=post_labels)
        
        return {
            "val_loss": loss, 
            "val_number": len(post_outputs),
            "preds": outputs,
            "images": images,
            "labels": labels}

    def validation_epoch_end(self, outputs):
        val_loss, num_val_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_val_items += output["val_number"]
        
        mean_val_loss = torch.tensor(val_loss / num_val_items)

        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        wandb_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch

        # qualitative results on wandb
        for output in outputs:
            fig = visualize(preds=output["preds"], imgs=output["images"], gts=output["labels"], num_slices=5)
            wandb.log({"Validation Output Visualizations": fig})
            plt.close()

        print(
            f"Current epoch: {self.current_epoch}"
            f"\nCurrent Mean Dice: {mean_val_dice:.4f}"
            f"\nBest Mean Dice: {self.best_val_dice:.4f} at Epoch: {self.best_val_epoch}"
            f"\n----------------------------------------------------")
        
        self.metric_values.append(mean_val_dice)

        # log on to wandb
        self.log_dict(wandb_logs)
        
        return {"log": wandb_logs}

    def test_step(self, batch, batch_idx):
        test_input1, test_input2 = batch["image_ses1"], batch["image_ses2"]
        test_inputs = torch.cat([test_input1, test_input2], dim=1)
        roi_size = self.inference_roi_size
        sw_batch_size = 4
        batch["pred"] = sliding_window_inference(test_inputs, roi_size, sw_batch_size, self.forward, overlap=0.5)

        post_test_out = [self.test_post_pred(i) for i in decollate_batch(batch)]
        # print(post_test_out[0]['pred'].shape)
        # print(post_test_out[0]['label_c'].shape)
        
        # make sure that the shapes of prediction and GT label are the same
        assert post_test_out[0]['pred'].shape == post_test_out[0]['label_c'].shape

        # Run ANIMA segmentation performance metrics on the predictions
        # NOTE: We use certain additional arguments below with the following purposes:
        #       -d -> surface distance eval, -l -> detection of lesions eval
        #       -a -> intra-lesion eval, -s -> segmentation eval, -X -> save as XML file
        seg_perf_analyzer_cmd = '%s -i %s -r %s -o %s -d -l -a -s -X'
        # subject_name = post_test_out[0]["label_c_meta_dict"]["filename_or_obj"] .strip(os.path.join(self.results_dir, args.model))
        subject_name = os.path.split(post_test_out[0]["label_c_meta_dict"]["filename_or_obj"])[1][:-18]
        # print(f"subject_name: {subject_name}")
        pred_file_name = f"{subject_name}_pred.nii.gz"
        gt_file_name = f"{subject_name}_gt.nii.gz"
        os.system(seg_perf_analyzer_cmd %
                    (os.path.join(self.anima_binaries_path, 'animaSegPerfAnalyzer'),
                    os.path.join(self.args.results_dir, args.model, args.best_model_name[:-5], subject_name, pred_file_name),
                    os.path.join(self.args.results_dir, args.model, args.best_model_name[:-5], subject_name, gt_file_name),
                    os.path.join(self.args.results_dir, args.model, args.best_model_name[:-5], subject_name , subject_name)))

        return {"subject_name": subject_name}

    def test_epoch_end(self, outputs):
        # Get all XML filepaths where ANIMA performance metrics
        path = os.path.join(self.args.results_dir, args.model, args.best_model_name[:-5])
        subject_filepaths = []
        for output in outputs:
            subject = output['subject_name']
            for f in os.listdir(os.path.join(path, subject)):
                if f.endswith('.xml'):
                    subject_filepaths.append(os.path.join(path, subject, f))
                
        test_metrics = defaultdict(list)
        # Update the test metrics dictionary by iterating over all subjects
        with open(os.path.join(path, 'test_metrics.txt'), 'a') as f:
            for subject_filepath in subject_filepaths:
                subject = os.path.split(subject_filepath)[-1].split('_')[0]
                root_node = ET.parse(source=subject_filepath).getroot()

                # In ANIMA 4.1.1, the metrics themselves are not computed when the GT is empty. 
                if not len(list(root_node)) == 13:  # because 13 metrics are computed in total
                    print('Skipping Subject=%s ENTIRELY Due to Empty GT!' % subject, file=f)
                    continue

                for metric in list(root_node):
                    name, value = metric.get('name'), float(metric.text)
                    if np.isinf(value) or np.isnan(value):
                        print('Skipping Metric=%s for Subject=%s Due to INF or NaNs!' % (name, subject))
                        continue

                    test_metrics[name].append(value)
        
            # Print aggregation of each metric via mean and standard dev.
            print('\n-------------- Test Phase Metrics [ANIMA v4.1.1] ----------------', file=f)
            for key in test_metrics:
                print('\t%s -> Mean: %0.4f Std: %0.2f' % (key, np.mean(test_metrics[key]), np.std(test_metrics[key])), file=f)
            print('-----------------------------------------------------------------', file=f)
        
        print("------- Testing Done! -------")

def main(args):
    # Setting the seed
    pl.seed_everything(args.seed)

    dataset_root = "/home/GRAMES.POLYMTL.CA/u114716/datasets/ms-challenge-2021_preprocessed_clean/"
    # dataset_root = "/home/GRAMES.POLYMTL.CA/u114716/duke/temp/muena/ms-challenge-2021_preprocessed/data_processed_clean/"

    save_path = args.save_path
    # save_dir = "/home/GRAMES.POLYMTL.CA/u114716/sci-zurich_project/modeling/saved_models"
    results_dir = args.results_dir

    # define models
    # TODO: add options for more models
    if args.model in ["unet", "UNet"]:            
        net = UNet(
            spatial_dims=3, in_channels=2, out_channels=2,
            channels=(
                args.init_filters, 
                args.init_filters * 2, 
                args.init_filters * 4, 
                args.init_filters * 8, 
                args.init_filters * 16),
            strides=(2, 2, 2, 2),
            num_res_units=2,)
        exp_id =f"ms-ch-21_{args.model}_{args.loss_func}_{args.optimizer}_lr={args.learning_rate}" \
                f"_initf={args.init_filters}_ps={args.patch_size}"
    elif args.model in ["unetr", "UNETR"]:
        net = UNETR(
            in_channels=2, out_channels=2, 
            img_size=((args.patch_size,) * 3),
            feature_size=args.feature_size, 
            hidden_size=args.hidden_size, 
            mlp_dim=args.mlp_dim, 
            num_heads=args.num_heads,
            pos_embed="perceptron", 
            norm_name="instance", 
            res_block=True, dropout_rate=0.0,)
        exp_id =f"ms-ch-21_{args.model}_{args.loss_func}_{args.optimizer}_lr={args.learning_rate}" \
                f"_fs={args.feature_size}_hs={args.hidden_size}_mlpd={args.mlp_dim}_nh={args.num_heads}" \
                f"_ps={args.patch_size}"
    elif args.model in ["segresnet", "SegResNet"]:
        net = SegResNet(
            in_channels=2, out_channels=2,
            init_filters=args.init_filters,
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            dropout_prob=0.2,)
        exp_id =f"ms-ch-21_{args.model}_{args.loss_func}_{args.optimizer}_lr={args.learning_rate}" \
                f"_initf={args.init_filters}_ps={args.patch_size}"

    # Define the loss function and the optimizer
    ce_weights = torch.FloatTensor([0.001, 0.999])  # for bg, fg
    if args.loss_func == "dice_ce":
        loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=ce_weights)
    elif args.loss_func == "dice_ce_sq":
        loss_function = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True, ce_weight=ce_weights, squared_pred=True)
    elif args.loss_func == "dice":
        loss_function = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)    # because there are 2 classes (bg=0 and fg=1)
    else:
        loss_function = FocalLoss(include_background=False, to_onehot_y=True, gamma=2.0)

    if args.optimizer in ["adamw", "AdamW", "Adamw"]:
        optimizer_class = torch.optim.AdamW
    elif args.optimizer in ["SGD", "sgd"]:
        optimizer_class = torch.optim.SGD

    if not args.only_eval:

        for fold_n in range(args.num_cv_folds):
            print(f"-------- Going over fold-{fold_n} of the {args.dataset} dataset! --------")
            # instantiate the PL model
            pl_model = Model(args, fold_num=fold_n, data_root=dataset_root, net=net, loss_function=loss_function, optimizer_class=optimizer_class)
    
            # exp_id = exp_id + f"_fold-{fold_n}"
            timestamp = datetime.now().strftime(f"%Y%m%d-%H%M%S")   # prints in YYYYMMDD-HHMMSS format
            save_exp_id = exp_id + f"_fold-{fold_n}_{timestamp}"
            wandb_logger = pl.loggers.WandbLogger(
                                name=save_exp_id,
                                group=args.model, 
                                log_model=True, # save best model using checkpoint callback
                                project=args.dataset,
                                entity='naga-karthik',
                                config=args)

            # to save the best model on validation
            if not os.path.exists(os.path.join(save_path, args.model)):
                os.makedirs(os.path.join(save_path, args.model), exist_ok=True)
            checkpoint_callback = pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(save_path, args.model), filename=save_exp_id, monitor='val_loss', 
                save_top_k=1, mode="min", save_last=False, save_weights_only=True)
            
            lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
            
            early_stopping = pl.callbacks.EarlyStopping(
                monitor="val_loss", min_delta=0.00,
                patience=args.patience, verbose=False, mode="min")

            # initialise Lightning's trainer.
            trainer = pl.Trainer(
                devices=args.num_gpus, accelerator="gpu", strategy="ddp",
                logger=wandb_logger, 
                callbacks=[checkpoint_callback, lr_monitor, early_stopping],
                check_val_every_n_epoch=args.check_val_every_n_epochs,
                max_epochs=args.max_epochs, 
                precision=32,
                enable_progress_bar=args.enable_progress_bar)

            # Train!
            trainer.fit(pl_model)
            print("------- Training Done! -------")
            
            print("------- Printing the Best Model Path! ------") 
            # print best checkpoint after training
            print(trainer.checkpoint_callback.best_model_path)

            # closing the current wandb instance so that a new one is created for the next fold
            wandb.finish()
        
    else:
        # the inference is done on the cpu, to test the model run the following:
        # "python main_pl.py -e -m unetr -bmn <file-name-ending-with .ckpt> -tfn <fold-num-as-seen-in-best-model-name>"
        print("------- Loading the Best Model! ------")
        # load the best checkpoint after training
        f_num = args.test_fold_num
        best_model_name = args.best_model_name
        pl_model = Model(args, fold_num=f_num, data_root=dataset_root, net=net, loss_function=loss_function, optimizer_class=optimizer_class)
        loaded_model = pl_model.load_from_checkpoint(os.path.join(args.save_path, args.model, best_model_name), strict=False)
        loaded_model.eval()

        trainer = pl.Trainer(devices=1, accelerator="cpu", max_epochs=args.max_epochs, precision=32, 
            enable_progress_bar=True, limit_train_batches=0, limit_val_batches=0)
        trainer.fit(model=pl_model)

        print("------- Testing Begins! -------")
        trainer.test(loaded_model)


def get_ce_weights(label):    
    '''
    label/target: shape - [C, D, H, W]  
    '''
    if label.shape[0] > 1:
        label = label[1:]  # for One-Hot format data, remove the background channel
    
    label_flat = torch.ravel(torch.any(label,0))  # in case label has multiple dimensions
    num_fg_indices = torch.nonzero(label_flat).shape[0]
    num_bg_indices = torch.nonzero(~label_flat).shape[0]

    counts = torch.FloatTensor([num_bg_indices, num_fg_indices])
    if num_fg_indices == 0:
        counts[1] = 1e-5    # to prevent division by zero
        # and discard the loss coming only from the bg_indices
    
    ce_weights = 1.0/counts
    norm_ce_weights = ce_weights/ce_weights.sum()

    return norm_ce_weights.cuda()

def visualize(preds, imgs, gts, num_slices=10):
    # getting ready for post processing
    imgs, gts = imgs[:, 1].detach().cpu(), gts.detach().cpu(),     # plotting ses-02 flair image
    imgs = imgs.squeeze(dim=1).numpy()  
    gts = gts.squeeze(dim=1)
    preds = torch.argmax(preds, dim=1).detach().cpu()

    fig, axs = plt.subplots(3, num_slices, figsize=(9, 9))
    fig.suptitle('Original --> Ground Truth --> Prediction')
    slice = imgs.shape[3]//2 + 60
    slice_nums = np.array([(slice-15), (slice-5), (slice), (slice+5), (slice+15)])

    for i in range(num_slices):
        axs[0, i].imshow(imgs[0, :, :, slice_nums[i]].T, cmap='gray'); axs[0, i].axis('off') 
        axs[1, i].imshow(gts[0, :, :, slice_nums[i]].T); axs[1, i].axis('off')
        axs[2, i].imshow(preds[0, :, :, slice_nums[i]].T); axs[2, i].axis('off')
        # axs[0, i].imshow(imgs[0, :, slice_nums[i], :].T, cmap='gray'); axs[0, i].axis('off')    # coronal
        # axs[0, i].imshow(imgs[0, slice_nums[i], :, :].T, cmap='gray'); axs[0, i].axis('off')  # sagittal


    plt.tight_layout()
    fig.show()
    return fig



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Script for training custom models for SCI Lesion Segmentation.')
    # Arguments for model, data, and training and saving
    parser.add_argument('-e', '--only_eval', default=False, action='store_true', help='Only do evaluation, i.e. skip training!')
    parser.add_argument('-m', '--model', 
                        choices=['unet', 'UNet', 'unetr', 'UNETR', 'segresnet', 'SegResNet'], 
                        default='unet', type=str, help='Model type to be used')
    # dataset
    parser.add_argument("--dataset", type=str, default='ms-challenge-2021', help="dataset to be used.")
    parser.add_argument('-ncv', '--num_cv_folds', default=5, type=int, help="k for performing k-fold cross validation")
    parser.add_argument('-ngts', '--num_gts', choices=[1, 5], default=1, type=int, 
                        help="Number of GTs to use - 1 for consensus GT only; 5 for 1 consensus + 4 other raters")
    parser.add_argument('-nspv', '--num_samples_per_volume', default=4, type=int, help="Number of samples to crop per volume")    
    
    # unet model 
    # parser.add_argument('-t', '--task', choices=['sc', 'mc'], default='sc', type=str, help="Single-channel or Multi-channel model ")
    parser.add_argument('-initf', '--init_filters', default=16, type=int, help="Number of Filters in Init Layer")
    # parser.add_argument('-ccs', '--center_crop_size', nargs='+', default=[128, 256, 96], help='List containing center crop size for preprocessing')
    parser.add_argument('-ps', '--patch_size', type=int, default=128, help='List containing subvolume size')
    # parser.add_argument('-srs', '--stride_size', nargs='+', default=[128, 256, 96], help='List containing stride size')

    # unetr model 
    parser.add_argument('-fs', '--feature_size', default=16, type=int, help="Feature Size")
    parser.add_argument('-hs', '--hidden_size', default=768, type=int, help='Dimensionality of hidden embeddings')
    parser.add_argument('-mlpd', '--mlp_dim', default=2048, type=int, help='Dimensionality of MLP layer')
    parser.add_argument('-nh', '--num_heads', default=12, type=int, help='Number of heads in Multi-head Attention')
    
    # optimizations
    parser.add_argument('-lf', '--loss_func',
                         choices=['dice', 'dice_ce', 'dice_f'],
                         default='dice', type=str, help="Loss function to use")
    parser.add_argument('-gpus', '--num_gpus', default=1, type=int, help="Number of GPUs to use")
    parser.add_argument('-nw', '--num_workers', default=4, type=int, help='Number of workers for the dataloaders')
    parser.add_argument('-p', '--precision', default=32, type=int, help='Precision for training')
    parser.add_argument('-me', '--max_epochs', default=1000, type=int, help='Number of epochs for the training process')
    parser.add_argument('-bs', '--batch_size', default=2, type=int, help='Batch size of the training and validation processes')
    parser.add_argument('-opt', '--optimizer', 
                        choices=['adamw', 'AdamW', 'SGD', 'sgd'], 
                        default='adamw', type=str, help='Optimizer to use')
    parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='Learning rate for training the model')
    parser.add_argument('-pat', '--patience', default=200, type=int, help='number of validation steps (val_every_n_iters) to wait before early stopping')
    parser.add_argument('--T_0', default=100, type=int, help='number of steps in each cosine cycle')
    parser.add_argument('-epb', '--enable_progress_bar', default=False, action='store_true', help='by default is disabled since it doesnt work in colab')
    parser.add_argument('-cve', '--check_val_every_n_epochs', default=30, type=int, help='num of epochs to wait before validation')
    # saving
    parser.add_argument('-sp', '--save_path', 
                        default=f"/home/GRAMES.POLYMTL.CA/u114716/ms-challenge-2021/saved_models", 
                        type=str, help='Path to the saved models directory')
    parser.add_argument('-c', '--continue_from_checkpoint', default=False, action='store_true', help='Load model from checkpoint and continue training')
    parser.add_argument('-se', '--seed', default=42, type=int, help='Set seeds for reproducibility')
    # testing
    parser.add_argument('-rd', '--results_dir', 
                    default=f"/home/GRAMES.POLYMTL.CA/u114716/ms-challenge-2021/model_predictions", 
                    type=str, help='Path to the model prediction results directory')
    parser.add_argument('-bmn', '--best_model_name', 
                        default=f"/home/GRAMES.POLYMTL.CA/u114716/ms-challenge-2021/saved_models", 
                        type=str, help='Name of best .ckpt file to load for testing')
    parser.add_argument('-tfn', '--test_fold_num', default=2, type=int, help='Fold num found in the best model file name')




    args = parser.parse_args()

    main(args)
