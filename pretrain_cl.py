#!/usr/bin/env python
from __future__ import print_function

import argparse
from cv2 import log
import yaml
from easydict import EasyDict as edict  
import os
import random
from os.path import join, isfile
from os import makedirs
from datetime import datetime
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import DataLoader

from orissl_cvm import PACKAGE_ROOT_DIR
from orissl_cvm.augmentations import SimSiamTransform
from orissl_cvm.models import get_backbone
from orissl_cvm.tools.val_cvm import val
from orissl_cvm.tools import save_checkpoint, create_logger, log_config_to_file, visualize
from orissl_cvm.augmentations import input_transform
from orissl_cvm.datasets.cvact_dataset import CVACTDataset, ImagePairsFromList, ImagesFromList
from orissl_cvm.models import get_model
from orissl_cvm.optimizers import get_optimizer, LR_Scheduler

from tqdm.auto import trange, tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrain the encoder by contrastive learning')
    parser.add_argument('--config_path', type=str, default=join(PACKAGE_ROOT_DIR, os.pardir, 'configs/pretrain_cl.yaml'),
                        help='File name (with extension) to the yaml file that stores the configuration')

    # Parse arguments
    opt = parser.parse_args()

    # Load config file, create logger
    cfg_file = opt.config_path
    assert os.path.isfile(cfg_file)
    with open(cfg_file, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.Loader))
    cfg.identifier = f'{cfg.model.name}_{cfg.model.backbone}_{cfg.model.pool}'
    if 'shared' in cfg.model:
        cfg.identifier += '_shared' if cfg.model.shared else '_noshared'
    if 'category' in cfg.model:
        cfg.identifier += f'_{cfg.model.category}'
    cfg.identifier += f'_{cfg.dataset.name}_{cfg.dataset.dataset_version}'

    logdir = join(cfg.train.save_path, datetime.now().strftime('%b%d-%H%M') + '_' + cfg.identifier)
    makedirs(logdir)
    shutil.copyfile(cfg_file, join(logdir, cfg_file.split('/')[-1]))
    log_file = join(logdir, 'log_train.txt')
    logger = create_logger(log_file, rank=0) # NOTE only 1 gpu
    logger.info('**********************Configs**********************')
    log_config_to_file(cfg, logger=logger)

    logger.info('**********************Start logging**********************')
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.train.gpu_ids
    os.environ["MKL_NUM_THREADS"] = cfg.train.threads
    os.environ["NUMEXPR_NUM_THREADS"] = cfg.train.threads
    os.environ["OMP_NUM_THREADS"] = cfg.train.threads
    logger.info('CUDA_VISIBLE_DEVICES=%s' % os.environ["CUDA_VISIBLE_DEVICES"])

    # CUDA setting
    cuda = not cfg.train.no_cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda" if cuda else "cpu")
    logger.info(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
    logger.info(f'torch.cuda.current_device(): {torch.cuda.current_device()}')

    # Random seeds
    if cfg.train.seed is not None:
        logger.info(f"Deterministic with seed = {cfg.train.seed}")
        random.seed(cfg.train.seed) 
        np.random.seed(cfg.train.seed) 
        torch.manual_seed(cfg.train.seed)
        if cuda:
            torch.cuda.manual_seed(cfg.train.seed)
            torch.backends.cudnn.deterministic = True 
            torch.backends.cudnn.benchmark = False 

    # Model: with resuming or not
    logger.info('===> Building model')

    if cfg.train.resume_path: # if already started training earlier and continuing
        if isfile(cfg.train.resume_path):
            logger.info("===> loading checkpoint '{}'".format(cfg.train.resume_path))
            checkpoint = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage)

            model = get_model(cfg.model)
            model.load_state_dict(checkpoint['state_dict'])
            cfg.train.start_epoch = checkpoint['epoch']

            logger.info("===> loaded checkpoint '{}'".format(cfg.train.resume_path))
        else:
            raise FileNotFoundError("===> no checkpoint found at '{}'".format(cfg.train.resume_path))
    else: # if not, assume fresh training instance and will initially generate cluster centroids
        logger.info('===> Loading model')
        model = get_model(cfg.model)

    model = model.to(device)
    logger.info(model)

    # If DataParallel
    # TODO learn more about multi-gpu training. Actually more stuff needs
    # to be considered, e.g., only log info of local rank 0
    isParallel = False
    if cfg.train.n_gpu > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        isParallel = True

    # Dataset and dataloader
    logger.info('===> Loading dataset(s)')
    train_dataset = CVACTDataset(cfg.dataset.dataset_root_dir,
                                 mode='train', 
                                 transform=None,
                                 logger=logger,
                                 version=cfg.dataset.dataset_version)
    logger.info(f'Full num of image pairs in training set: {train_dataset.qImages.shape[0]}')
    logger.info(f'Num of queries in training set: {len(train_dataset)}')
    if cfg.model.shared:
        train_dataset = ImagePairsFromList(train_dataset.root_dir, 
                                    train_dataset.qImages, 
                                    transform=SimSiamTransform((cfg.model.img_size_h, cfg.model.img_size_w)))
        train_dataloader = DataLoader(dataset=train_dataset, 
            num_workers=cfg.dataset.n_workers,
            batch_size=cfg.train.batch_size, 
            shuffle=True,
            collate_fn=ImagePairsFromList.collate_fn,
            pin_memory=cuda,
            drop_last=True
        )
    else:
        train_dataset = ImagesFromList(train_dataset.root_dir, 
                                    train_dataset.qImages, 
                                    transform=SimSiamTransform((cfg.model.img_size_h, cfg.model.img_size_w)), 
                                    category=cfg.model.category)
        train_dataloader = DataLoader(dataset=train_dataset, 
            num_workers=cfg.dataset.n_workers,
            batch_size=cfg.train.batch_size, 
            shuffle=True,
            collate_fn=ImagesFromList.collate_fn,
            pin_memory=cuda,
            drop_last=True
        )

    # NOTE visualize batches for debug
    # visualize_dataloader(training_data_loader)

    # Optimizer and scheduler
    optimizer = None
    scheduler = None
    optimizer = get_optimizer(
        cfg.hyperparams.optim, model, 
        lr=cfg.hyperparams.base_lr*cfg.train.batch_size/256, 
        momentum=cfg.hyperparams.momentum,
        weight_decay=cfg.hyperparams.weight_decay)
    scheduler = LR_Scheduler(
        optimizer,
        cfg.hyperparams.warmup_epochs, cfg.hyperparams.warmup_lr*cfg.train.batch_size/256, 
        cfg.train.n_epochs, cfg.hyperparams.base_lr*cfg.train.batch_size/256, cfg.hyperparams.final_lr*cfg.train.batch_size/256, 
        len(train_dataloader),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )
    if cfg.train.resume_path:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    # SummaryWriter, and create logdir
    writer = SummaryWriter(log_dir=logdir)
    cfg.train.save_file_path = join(logdir, 'checkpoints')
    makedirs(cfg.train.save_file_path)

    not_improved = 0
    best_score = 0
    if cfg.train.resume_path:
        not_improved = checkpoint['not_improved']
        best_score = checkpoint['best_score']

    # Training
    logger.info('===> Training model')
    for epoch in trange(cfg.train.start_epoch + 1, cfg.train.n_epochs + 1, desc='Epoch number'.rjust(15), position=0):
        epoch_loss = 0
        n_batches = (len(train_dataset) + cfg.train.batch_size - 1) // cfg.train.batch_size
        model.train()
        local_progress = tqdm(train_dataloader, position=1, leave=False, desc='Train Iter'.rjust(15))

        for iteration, batch in enumerate(local_progress):
            if batch is None:
                continue
            if cfg.model.shared:
                (im1, im2), (im3, im4), labels = batch # TODO `labels` makes no sense!
                model.zero_grad()
                data_dict1 = model.forward(im1.to(device, non_blocking=True), im2.to(device, non_blocking=True))
                data_dict3 = model.forward(im3.to(device, non_blocking=True), im4.to(device, non_blocking=True))
                loss = data_dict1['loss'].mean() + data_dict3['loss'].mean()
            else:
                (im1, im2), labels = batch # TODO `labels` makes no sense!
                model.zero_grad()
                data_dict = model.forward(im1.to(device, non_blocking=True), im2.to(device, non_blocking=True))
                loss = data_dict['loss'].mean() # ddp

            # NOTE for debug
            # visualize.visualize_assets(im1, im2, im3, im4)
            loss.backward()
            optimizer.step()
            scheduler.step()
            # data_dict.update({'lr': scheduler.get_lr()})

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % (n_batches // 5) == 0 or n_batches <= 10:
                tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration,
                                                                        n_batches, batch_loss))
                writer.add_scalar('Train/Loss', batch_loss, ((epoch - 1) * n_batches) + iteration)
                writer.add_scalar('Train/lr', scheduler.get_lr(), ((epoch - 1) * n_batches) + iteration)

        avg_loss = epoch_loss / n_batches
        tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss))
        writer.add_scalar('Train/AvgLoss', avg_loss, epoch)

        is_best = avg_loss < best_score
        if is_best:
            not_improved = 0
            best_score = avg_loss
        else:
            not_improved += 1

        if (epoch % cfg.train.eval_every) == 0:
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_score': best_score,
                'not_improved': not_improved,
                'optimizer': optimizer.state_dict(),
                'parallel': isParallel,
            }, cfg, is_best)

    writer.close()
    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs
    logger.info('Done')