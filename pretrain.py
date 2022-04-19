#!/usr/bin/env python

'''
MIT License

Copyright (c) 2021 Stephen Hausler, Sourav Garg, Ming Xu, Michael Milford and Tobias Fischer

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Significant parts of our code are based on [Nanne's pytorch-netvlad repository]
(https://github.com/Nanne/pytorch-NetVlad/), as well as some parts from the [Mapillary SLS repository]
(https://github.com/mapillary/mapillary_sls)

Modified by Jianfeng Cui
'''


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
from orissl_cvm.tools.pretrain_epoch import pretrain_epoch
from orissl_cvm.tools.val_cls import val_cls
from orissl_cvm.tools import save_checkpoint, create_logger, log_config_to_file
from orissl_cvm.utils import input_transform
from orissl_cvm.datasets.cvact_dataset import CVACTDataset
from orissl_cvm.models.safa import SAFAvgg16Cls
from orissl_cvm.tools.visualize import visualize_dataloader

from tqdm.auto import trange


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--config_path', type=str, default=join(PACKAGE_ROOT_DIR, os.pardir, 'configs/train.ini'),
                        help='File name (with extension) to an ini file that stores most of the configuration data')

    # Parse arguments
    opt = parser.parse_args()

    # Load config file, create logger
    cfg_file = opt.config_path
    assert os.path.isfile(cfg_file)
    with open(cfg_file, 'r') as f:
        cfg = edict(yaml.load(f, Loader=yaml.Loader))

    logdir = join(cfg.train.save_path, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + cfg.train.identifier)
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
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if cuda:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(cfg.train.seed)

    # Model: with resuming or not
    logger.info('===> Building model')

    if cfg.train.resume_path: # if already started training earlier and continuing
        if isfile(cfg.train.resume_path):
            logger.info("===> loading checkpoint '{}'".format(cfg.train.resume_path))
            checkpoint = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage)

            model = SAFAvgg16Cls()

            model.load_state_dict(checkpoint['state_dict'])
            cfg.train.start_epoch = checkpoint['epoch']

            logger.info("===> loaded checkpoint '{}'".format(cfg.train.resume_path))
        else:
            raise FileNotFoundError("===> no checkpoint found at '{}'".format(cfg.train.resume_path))
    else: # if not, assume fresh training instance and will initially generate cluster centroids
        logger.info('===> Loading model')
        model = SAFAvgg16Cls()

    logger.info(model)

    # If DataParallel
    # TODO learn more about multi-gpu training. Actually more stuff needs
    # to be considered, e.g., only log info of local rank 0
    isParallel = False
    if cfg.train.n_gpu > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        isParallel = True

    # Optimizer and scheduler
    optimizer = None
    scheduler = None

    if cfg.params.optim == 'ADAM':
        optimizer = optim.Adam(filter(lambda par: par.requires_grad,
                                      model.parameters()), lr=cfg.params.lr)  # , betas=(0,0.9))
    elif cfg.params.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda par: par.requires_grad,
                                     model.parameters()), lr=cfg.params.lr,
                              momentum=cfg.params.momentum,
                              weight_decay=cfg.params.weight_decay)

        # TODO include scheduler later
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.train.lr_step,
        #                                       gamma=config.train.lr_gamma)
    else:
        raise ValueError('Unknown optimizer: ' + cfg.params.optim)
    
    model = model.to(device)

    if cfg.train.resume_path:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Loss
    # TODO delete it later, because we're actually using our own loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Dataset and dataloader
    logger.info('===> Loading dataset(s)')
    train_dataset = CVACTDataset(cfg.train.dataset_root_dir,
                                 mode='train', 
                                 transform=input_transform(),
                                 logger=logger,
                                 task='ssl')
    logger.info(f'Full num of image pairs in training set: {train_dataset.qImages.shape[0]}')
    logger.info(f'Num of queries in training set: {len(train_dataset)}')

    training_data_loader = DataLoader(dataset=train_dataset, 
        num_workers=cfg.train.n_workers,
        batch_size=cfg.train.batch_size, 
        shuffle=False,
        collate_fn = train_dataset.collate_fn_ssl, 
        pin_memory=cuda
    )

    # NOTE visualize batches for debug
    # visualize_dataloader(training_data_loader)

    # validation_dataset = CVACTDataset(cfg.train.dataset_root_dir, 
    #                                   mode='val', 
    #                                   transform=input_transform(),
    #                                   logger=logger,
    #                                   task='ssl')
    # NOTE for debug, use train set it self to validate
    validation_dataset = train_dataset
    val_data_loader = DataLoader(dataset=validation_dataset, 
        num_workers=cfg.train.n_workers,
        batch_size=cfg.train.batch_size, 
        shuffle=False,
        collate_fn=validation_dataset.collate_fn_ssl, 
        pin_memory=cuda
    )
    logger.info(f'Full num of image pairs in validation set: {validation_dataset.qImages.shape[0]}')
    logger.info(f'Num of queries in validation set: {len(validation_dataset)}')

    # SummaryWriter, and create logdir
    writer = SummaryWriter(log_dir=logdir)
    cfg.train.save_file_path = join(logdir, 'checkpoints')
    makedirs(cfg.train.save_file_path)

    not_improved = 0
    best_score = 0
    if cfg.train.resume_path:
        not_improved = checkpoint['not_improved']
        best_score = checkpoint['best_score']

    #
    # Training
    #
    logger.info('===> Training model')
    # TODO load from a model. delete it later
    # model.load_state_dict(torch.load("work_dirs/Apr04_21-49-08_cvact_resnet50/checkpoints/a_temp_for_debug.pth"))
    
    for epoch in trange(cfg.train.start_epoch + 1, cfg.train.n_epochs + 1, desc='Epoch number'.rjust(15), position=0):

        pretrain_epoch(train_dataset, training_data_loader, model, optimizer, criterion, device, epoch, cfg, writer)

        # TODO delete it later
        # torch.save(model.state_dict(), join(opt.save_file_path, "a_temp_for_debug.pth"))

        if scheduler is not None:
            # TODO a little out-of-date. Use scheduler.step() after optimizer.step() later
            scheduler.step(epoch)

        if (epoch % cfg.train.eval_every) == 0:
            acc = val_cls(val_data_loader, model, device, cfg, writer, epoch,
                          write_tboard=True, pbar_position=1)
            is_best = acc > best_score
            if is_best:
                not_improved = 0
                best_score = acc
            else:
                not_improved += 1

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'accuracy': acc,
                'best_score': best_score,
                'not_improved': not_improved,
                'optimizer': optimizer.state_dict(),
                'parallel': isParallel,
            }, cfg, is_best)

            if cfg.train.patience > 0 and not_improved > (cfg.train.patience / cfg.train.eval_every):
                logger.info('Performance did not improve for', cfg.train.patience, 'epochs. Stopping.')
                break

    logger.info("=> Best accuracy: {:.4f}".format(best_score), flush=True)
    writer.close()

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs

    logger.info('Done')