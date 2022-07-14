#!/usr/bin/env python
from __future__ import print_function

import argparse
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

from clsslcvm import PACKAGE_ROOT_DIR
from clsslcvm.augmentations import SimSiamTransform
from clsslcvm.models import get_backbone
from clsslcvm.tools.val_cvm import val
from clsslcvm.tools import save_checkpoint, create_logger, log_config_to_file, \
                visualize, register_hook, show_cam_on_image, get_dataset, get_model_with_ckpt
from clsslcvm.augmentations import input_transform
from clsslcvm.datasets.generic_dataset import ImagesFromList
from clsslcvm.models import get_model
from clsslcvm.optimizers import get_optimizer, LR_Scheduler
from clsslcvm.tools.visualize import visualize_assets
from clsslcvm.tools import humanbytes
from clsslcvm.tools.logger import setup_logger

from loguru import logger
from tqdm.auto import trange, tqdm
#* Disable tqdm progress bar
from functools import partialmethod
tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


@logger.catch
def main(cfg, cfg_file):
    # logger
    logdir = join(cfg.train.save_path, datetime.now().strftime('%b%d-%H%M') + '_' + cfg.identifier)
    makedirs(logdir)
    shutil.copyfile(cfg_file, join(logdir, cfg_file.split('/')[-1]))
    setup_logger(
        logdir,
        distributed_rank=0,
        filename='log_train.txt',
        mode="a",
    )
    logger.info('**********************Config file**********************')
    log_config_to_file(cfg, logger=logger)
    logger.info('**********************Settings**********************')
    if not cfg.train.slurm:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.train.gpu_ids
        os.environ["MKL_NUM_THREADS"] = cfg.train.threads
        os.environ["NUMEXPR_NUM_THREADS"] = cfg.train.threads
        os.environ["OMP_NUM_THREADS"] = cfg.train.threads
        logger.info('CUDA_VISIBLE_DEVICES=%s' % os.environ["CUDA_VISIBLE_DEVICES"])

    # CUDA setting
    if not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")
    device = torch.device("cuda")
    logger.info(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
    logger.info(f'torch.cuda.current_device(): {torch.cuda.current_device()}')

    # Random seeds
    if cfg.train.seed is not None:
        logger.info(f"Deterministic with seed = {cfg.train.seed}")
        random.seed(cfg.train.seed) 
        np.random.seed(cfg.train.seed) 
        torch.manual_seed(cfg.train.seed)
        torch.cuda.manual_seed(cfg.train.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Model
    logger.info('===> Building model')
    model, checkpoint = get_model_with_ckpt(cfg, logger)
    model = model.to(device)
    logger.info(model)

    # If DataParallel
    isParallel = False
    if cfg.train.n_gpu > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        isParallel = True

    # Dataset and dataloader
    logger.info('===> Loading dataset(s)')
    train_set, train_set_queries_grd, train_set_queries_sat, \
        train_loader_queries_grd, train_loader_queries_sat = get_dataset(cfg, logger)
    if cfg.model.category == 'grd':
        train_set_queries = train_set_queries_grd
        train_loader_queries = train_loader_queries_grd
    elif cfg.model.category == 'sat':
        train_set_queries = train_set_queries_sat
        train_loader_queries = train_loader_queries_sat
    else:
        raise NotImplementedError

    # Optimizer and scheduler
    optimizer = get_optimizer(
        cfg.hyperparams.optim, model, 
        lr=cfg.hyperparams.base_lr*cfg.train.batch_size/256, 
        momentum=cfg.hyperparams.momentum,
        weight_decay=cfg.hyperparams.weight_decay)
    scheduler = LR_Scheduler(
        optimizer,
        cfg.hyperparams.warmup_epochs, cfg.hyperparams.warmup_lr*cfg.train.batch_size/256, 
        cfg.train.n_epochs, cfg.hyperparams.base_lr*cfg.train.batch_size/256, cfg.hyperparams.final_lr*cfg.train.batch_size/256, 
        len(train_set_queries),
        constant_predictor_lr=True # see the end of section 4.2 predictor
    )
    # if cfg.train.resume_path:
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    
    # SummaryWriter, and create logdir
    writer = SummaryWriter(log_dir=logdir)
    cfg.train.save_file_path = join(logdir, 'checkpoints')
    makedirs(cfg.train.save_file_path)
    not_improved = 0
    best_score = 0
    if cfg.train.resume_path:
        not_improved = checkpoint['not_improved']
        best_score = checkpoint['best_score']

    logger.info('Allocated: ' + humanbytes(torch.cuda.memory_allocated()) + \
                ', Cached: ' + humanbytes(torch.cuda.memory_reserved()))
    # Training
    logger.info('**********************Start training**********************')
    global_progress = trange(cfg.train.start_epoch + 1, cfg.train.n_epochs + 1, desc='Epoch number')
    for epoch in global_progress:

        batch_size = cfg.train.batch_size
        epoch_loss = 0
        n_batches = (len(train_set_queries) + batch_size - 1) // batch_size
        model.train()

        local_progress = tqdm(train_loader_queries, desc='Train Iter')
        for it, batch in enumerate(local_progress):
            (im1, im2), labels = batch
            model.zero_grad()
            data_dict = model.forward(im1.to(device, non_blocking=True), im2.to(device, non_blocking=True))
            loss = data_dict['loss'].mean() # ddp
            loss.backward()
            optimizer.step()
            scheduler.step()
            data_dict.update({'lr':scheduler.get_lr()})
            local_progress.set_postfix(data_dict)
            batch_loss = loss.item()
            epoch_loss += batch_loss
            if it == 0:
                logger.info('Allocated: ' + humanbytes(torch.cuda.memory_allocated()) + \
                            ', Cached: ' + humanbytes(torch.cuda.memory_reserved()))
            if it % (n_batches // 25) == 0 or n_batches <= 10:
                logger.info("Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, it, n_batches, batch_loss))
                writer.add_scalar('Train/Loss', batch_loss, ((epoch - 1) * n_batches) + it)
                writer.add_scalar('Train/lr', scheduler.get_lr(), ((epoch - 1) * n_batches) + it)

        avg_loss = epoch_loss / n_batches
        logger.info("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, avg_loss))
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
            }, None, None, cfg, is_best)
    writer.close()
    torch.cuda.empty_cache()
    logger.info('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pretrain the encoder by contrastive learning')
    parser.add_argument('--config_path', type=str, default=join(PACKAGE_ROOT_DIR, os.pardir, 'configs/train_cl.yaml'),
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
    if 'category' in cfg.model and cfg.model.category != '':
        cfg.identifier += f'_{cfg.model.category}'
    cfg.identifier += f'_{cfg.dataset.name}_{cfg.dataset.version}'
    if cfg.train.extra_identifier != '': cfg.identifier += f'_{cfg.train.extra_identifier}'

    main(cfg, cfg_file)