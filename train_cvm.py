#!/usr/bin/env python
import argparse
import yaml
from easydict import EasyDict as edict  
import os
import sys
import random
from os.path import join
from os import makedirs
from datetime import datetime
import shutil

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
import numpy as np

from clsslcvm import PACKAGE_ROOT_DIR
from clsslcvm.tools.train_cvm_epoch import train_epoch
from clsslcvm.tools.val_cvm import val
from clsslcvm.tools import save_checkpoint, create_logger, log_config_to_file, get_model_with_ckpt, get_dataset
from clsslcvm.loss import SoftTripletLoss

from tqdm.auto import trange


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--config_path', type=str, default=join(PACKAGE_ROOT_DIR, os.pardir, 'configs/train_cvm.yaml'),
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
    cfg.identifier += f'_{cfg.dataset.name}_{cfg.dataset.version}'
    if cfg.train.extra_identifier != '': cfg.identifier += f'_{cfg.train.extra_identifier}'

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
    if not torch.cuda.is_available(): raise Exception("No GPU found")
    device = torch.device("cuda")
    logger.info(f'torch.cuda.device_count(): {torch.cuda.device_count()}')
    logger.info(f'torch.cuda.current_device(): {torch.cuda.current_device()}')

    # Random seeds
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    torch.cuda.manual_seed(cfg.train.seed) # noinspection PyUnresolvedReferences

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

    # Optimizer and scheduler
    optimizer = None
    scheduler = None
    if cfg.hyperparams.optim == 'adam':
        optimizer = optim.Adam(filter(lambda par: par.requires_grad,
                                      model.parameters()), lr=cfg.hyperparams.lr)  # , betas=(0,0.9))
    elif cfg.hyperparams.optim == 'sgd':
        optimizer = optim.SGD(filter(lambda par: par.requires_grad,
                                     model.parameters()), lr=cfg.hyperparams.lr,
                              momentum=cfg.hyperparams.momentum,
                              weight_decay=cfg.hyperparams.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.hyperparams.lr_step,
                                              gamma=cfg.hyperparams.lr_gamma)
    else:
        raise ValueError('Unknown optimizer: ' + cfg.hyperparams.optim)
    if cfg.train.resume_path:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Loss
    criterion = SoftTripletLoss()
    # Dataset and dataloader
    logger.info('===> Loading dataset(s)')
    dataloader = get_dataset(cfg, logger)

    # SummaryWriter, and create logdir
    writer = SummaryWriter(log_dir=logdir)
    cfg.train.save_file_path = join(logdir, 'checkpoints')
    makedirs(cfg.train.save_file_path)
    not_improved = 0
    best_score = 0
    if cfg.train.resume_path:
        not_improved = checkpoint['not_improved']
        best_score = checkpoint['best_score']

    # If only want to validate
    if cfg.train.only_val:
        val(dataloader, model, device, cfg, writer, epoch_num=0,write_tboard=False, pbar_position=0)
        writer.close()
        torch.cuda.empty_cache()
        logger.info('Done')
        sys.exit()

    # Training
    logger.info('===> Training model')
    for epoch in trange(cfg.train.start_epoch + 1, cfg.train.n_epochs + 1, desc='Epoch number'.rjust(15), position=0):
        train_epoch(dataloader, model, optimizer, scheduler, criterion, device, epoch, cfg, writer)
        
        if (epoch % cfg.train.eval_every) == 0:
            recalls = val(dataloader, model, device, cfg, writer, epoch, write_tboard=True, pbar_position=1)
            is_best = recalls[1] > best_score
            if is_best:
                not_improved = 0
                best_score = recalls[1]
            else:
                not_improved += 1

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'recalls': recalls,
                'best_score': best_score,
                'not_improved': not_improved,
                'optimizer': optimizer.state_dict(),
                'parallel': isParallel,
            }, cfg, is_best)

            if cfg.train.patience > 0 and not_improved > (cfg.train.patience / cfg.train.eval_every):
                logger.info(f"Performance did not improve for {cfg.train.patience} epochs. Stopping.")
                break

    logger.info("=> Best Recall@5: {:.4f}".format(best_score))
    writer.close()
    torch.cuda.empty_cache()
    logger.info('Done')