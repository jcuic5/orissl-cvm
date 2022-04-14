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
import yaml
from easydict import EasyDict as edict  
import os
import random
from os.path import join, isfile
from os import makedirs
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
import numpy as np
from torch.utils.data import DataLoader

from orissl_cvm import PACKAGE_ROOT_DIR
from orissl_cvm.tools.pretrain_epoch import pretrain_epoch
from orissl_cvm.tools.val_cls import val_cls
from orissl_cvm.tools import save_checkpoint
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
    print('===> Terminal arguments')
    print(opt)

    # Load config file
    cfg_file = opt.config_path
    assert os.path.isfile(cfg_file)
    with open(cfg_file, 'r') as f:
        config = edict(yaml.load(f, Loader=yaml.Loader))
    
    print('===> Config file')

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = config.train.gpu_ids
    os.environ["MKL_NUM_THREADS"] = config.train.threads
    os.environ["NUMEXPR_NUM_THREADS"] = config.train.threads
    os.environ["OMP_NUM_THREADS"] = config.train.threads

    # CUDA setting
    cuda = not config.train.no_cuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")
    print(f'num of cuda device: {torch.cuda.device_count()}')
    print(f'current cuda device: {torch.cuda.current_device()}')

    # Random seeds
    random.seed(config.train.seed)
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    if cuda:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(config.train.seed)

    # Model: with resuming or not
    print('===> Building model')

    if config.train.resume_path: # if already started training earlier and continuing
        if isfile(config.train.resume_path):
            print("=> loading checkpoint '{}'".format(config.train.resume_path))
            checkpoint = torch.load(config.train.resume_path, map_location=lambda storage, loc: storage)

            model = SAFAvgg16Cls()

            model.load_state_dict(checkpoint['state_dict'])
            config.train.start_epoch = checkpoint['epoch']

            print("=> loaded checkpoint '{}'".format(config.train.resume_path))
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(config.train.resume_path))
    else: # if not, assume fresh training instance and will initially generate cluster centroids
        print('===> Loading model')
        model = SAFAvgg16Cls()

    desc_dim = model.SAFAvgg16.desc_dim
    print("===> Model")
    print(model)

    # If DataParallel
    # TODO learn more about multi-gpu training. Actually more stuff needs
    # to be considered, e.g., only log info of local rank 0
    isParallel = False
    if config.train.n_gpu > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        isParallel = True

    # Optimizer and scheduler
    optimizer = None
    scheduler = None

    if config.params.optim == 'ADAM':
        optimizer = optim.Adam(filter(lambda par: par.requires_grad,
                                      model.parameters()), lr=config.params.lr)  # , betas=(0,0.9))
    elif config.params.optim == 'SGD':
        optimizer = optim.SGD(filter(lambda par: par.requires_grad,
                                     model.parameters()), lr=config.params.lr,
                              momentum=config.params.momentum,
                              weight_decay=config.params.weight_decay)

        # TODO include scheduler later
        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.train.lr_step,
        #                                       gamma=config.train.lr_gamma)
    else:
        raise ValueError('Unknown optimizer: ' + config.params.optim)
    
    model = model.to(device)

    if config.train.resume_path:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Loss
    # TODO delete it later, because we're actually using our own loss function
    criterion = nn.CrossEntropyLoss().to(device)
    
    # Dataset and dataloader
    print('===> Loading dataset(s)')
    train_dataset = CVACTDataset(config.train.dataset_root_dir, 
                                 mode='train', 
                                 transform=input_transform(),
                                 task='ssl')
    print(f'Full num of images in training set: {train_dataset.qImages.shape[0]}')
    print(f'Num of queries in training set: {len(train_dataset)}')

    training_data_loader = DataLoader(dataset=train_dataset, 
        num_workers=config.train.n_workers,
        batch_size=config.train.batch_size, 
        shuffle=True,# TODO for debug. switch it back later
        collate_fn=train_dataset.collate_fn_ssl, 
        pin_memory=cuda
    )

    # NOTE visualize batches for debug
    # visualize_dataloader(training_data_loader)

    validation_dataset = CVACTDataset(config.train.dataset_root_dir, 
                                      mode='val', 
                                      transform=input_transform(),
                                      task='ssl')
    # NOTE for debug, use train set it self to validate
    # validation_dataset = train_dataset
    val_data_loader = DataLoader(dataset=validation_dataset, 
        num_workers=config.train.n_workers,
        batch_size=config.train.batch_size, 
        shuffle=False,
        collate_fn=validation_dataset.collate_fn_ssl, 
        pin_memory=cuda
    )
    print('===> Training query set:', len(train_dataset.qIdx))
    print('===> Evaluating on val set, query count:', len(validation_dataset.qIdx))

    # SummaryWriter, and create logdir
    writer = SummaryWriter(
        log_dir=join(config.train.save_path, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + config.train.identifier))

    logdir = writer.file_writer.get_logdir()
    config.train.save_file_path = join(logdir, 'checkpoints')
    makedirs(config.train.save_file_path)

    not_improved = 0
    best_score = 0
    if config.train.resume_path:
        not_improved = checkpoint['not_improved']
        best_score = checkpoint['best_score']

    #
    # Training
    #
    print('===> Training model')
    # TODO load from a model. delete it later
    # model.load_state_dict(torch.load("work_dirs/Apr04_21-49-08_cvact_resnet50/checkpoints/a_temp_for_debug.pth"))
    
    for epoch in trange(config.train.start_epoch + 1, config.train.n_epochs + 1, desc='Epoch number'.rjust(15), position=0):

        pretrain_epoch(train_dataset, training_data_loader, model, optimizer, criterion, desc_dim, device, epoch, config, writer)

        # TODO delete it later
        # torch.save(model.state_dict(), join(opt.save_file_path, "a_temp_for_debug.pth"))

        if scheduler is not None:
            # TODO a little out-of-date. Use scheduler.step() after optimizer.step() later
            scheduler.step(epoch)

        if (epoch % config.train.eval_every) == 0:
            # NOTE for debugging use training_data_loader
            acc = val_cls(val_data_loader, model, desc_dim, device, config, writer, epoch,
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
            }, config, is_best)

            if config.train.patience > 0 and not_improved > (config.train.patience / config.train.eval_every):
                print('Performance did not improve for', config.train.patience, 'epochs. Stopping.')
                break

    print("=> Best Accuracy: {:.4f}".format(best_score), flush=True)
    writer.close()

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs

    print('Done')