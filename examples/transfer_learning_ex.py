# ------------------------------------------------------------------------------------
# minDALL-E
# Copyright (c) 2021 Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------

import os
import sys
import argparse
from typing import Optional
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.distributed import rank_zero_only

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dalle.models import ImageGPT


parser = argparse.ArgumentParser()

parser.add_argument('-d', '--config-downstream', type=str, default=None, required=True)
parser.add_argument('-u', '--path-upstream', type=str, default=None, required=True)
parser.add_argument('-r', '--result-path', type=str, default=None, required=True)
parser.add_argument('--imagenet-path', type=str, default=None, required=True)

parser.add_argument('--n-gpus', type=int, default=1)
parser.add_argument('--seed', type=int, default=0)


args = parser.parse_args()


class ImageLogger(Callback):
    def __init__(self):
        super().__init__()

    @rank_zero_only
    def log_img(self, pl_module, batch, current_epoch, split="train"):
        with torch.no_grad():
            images, labels = batch
            recons = pl_module.stage1(images)
            images = images.cpu()
            recons = recons.cpu()

            grid_org = (torchvision.utils.make_grid(images, nrow=8) + 1.0) / 2.0
            grid_rec = (torchvision.utils.make_grid(recons, nrow=8) + 1.0) / 2.0
            grid_rec = torch.clip(grid_rec, min=0, max=1)

            pl_module.logger.experiment.add_image(f"images_org/{split}", grid_org, global_step=current_epoch)
            pl_module.logger.experiment.add_image(f"images_rec/{split}", grid_rec, global_step=current_epoch)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0 and trainer.current_epoch < 5:
            self.log_img(pl_module, batch, current_epoch=trainer.current_epoch, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx == 0 and trainer.current_epoch < 5:
            self.log_img(pl_module, batch, current_epoch=trainer.current_epoch, split="test")


class ImageNetDataModule(pl.LightningDataModule):
    def __init__(self,
                 data_dir: Optional[str] = None,
                 image_resolution: int = 256,
                 train_batch_size: int = 2,
                 valid_batch_size: int = 32,
                 num_workers: int = 8):
        super().__init__()

        self.data_dir = data_dir
        self.image_resolution = image_resolution
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.num_workers = num_workers

        self.train_transform = transforms.Compose(
            [transforms.Resize(image_resolution),
             transforms.RandomCrop(image_resolution),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )
        self.valid_transform = transforms.Compose(
            [transforms.Resize(image_resolution),
             transforms.CenterCrop(image_resolution),
             transforms.ToTensor(),
             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]
        )

    def setup(self, stage=None):
        self.trainset = torchvision.datasets.ImageNet(root=self.data_dir, split='train', transform=self.train_transform)
        self.validset = torchvision.datasets.ImageNet(root=self.data_dir, split='val', transform=self.valid_transform)

    def train_dataloader(self):
        return DataLoader(self.trainset,
                          batch_size=self.train_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)

    def valid_dataloader(self):
        return DataLoader(self.validset,
                          batch_size=self.valid_batch_size,
                          num_workers=self.num_workers,
                          pin_memory=True)


def setup_callbacks(config):
    # Setup callbacks
    now = datetime.now().strftime('%d%m%Y_%H%M%S')
    result_path = os.path.join(args.result_path,
                               os.path.basename(args.config_downstream).split('.')[0],
                               now)
    ckpt_path = os.path.join(result_path, 'ckpt')
    log_path = os.path.join(result_path, 'log')

    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_path,
        filename="imagenet-clscond-gen-{epoch:02d}" if config.stage2.use_cls_cond else
                 "imagenet-uncond-gen-{epoch:02d}",
        every_n_epochs=config.experiment.save_ckpt_freq,
        save_weights_only=True,
        save_last=True
    )
    logger = TensorBoardLogger(log_path, name="iGPT")
    logger_img = ImageLogger()
    return checkpoint_callback, logger, logger_img


if __name__ == '__main__':
    pl.seed_everything(args.seed)

    # Build iGPT
    model, config = ImageGPT.from_pretrained(args.path_upstream, args.config_downstream)

    # Setup callbacks
    ckpt_callback, logger, logger_img = setup_callbacks(config)

    # Build data modules
    dataset = ImageNetDataModule(data_dir=args.imagenet_path,
                                 image_resolution=config.dataset.image_resolution,
                                 train_batch_size=config.experiment.local_batch_size,
                                 valid_batch_size=config.experiment.valid_batch_size,
                                 num_workers=16)
    dataset.setup()
    train_dataloader = dataset.train_dataloader()
    valid_dataloader = dataset.valid_dataloader()
    print(f"len(train_dataset) = {len(dataset.trainset)}")
    print(f"len(valid_dataset) = {len(dataset.validset)}")

    # Calculate how many batches are accumulated
    assert config.experiment.total_batch_size % (config.experiment.local_batch_size * args.n_gpus) == 0
    grad_accm_steps = config.experiment.total_batch_size // (config.experiment.local_batch_size * args.n_gpus)
    config.optimizer.max_steps = len(dataset.trainset) // config.experiment.total_batch_size * config.experiment.epochs

    # Build trainer
    trainer = pl.Trainer(max_epochs=config.experiment.epochs,
                         accumulate_grad_batches=grad_accm_steps,
                         gradient_clip_val=config.optimizer.grad_clip_norm,
                         precision=16 if config.experiment.use_amp else 32,
                         callbacks=[ckpt_callback, logger_img],
                         accelerator="gpu",
                         devices=args.n_gpus,
                         strategy="ddp",
                         logger=logger)
    trainer.fit(model, train_dataloader, valid_dataloader)
