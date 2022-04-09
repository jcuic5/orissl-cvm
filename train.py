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
'''


from __future__ import print_function

import argparse
import configparser
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

from orissl_cvm.utils import PACKAGE_ROOT_DIR
from orissl_cvm.utils.train_epoch import train_epoch
from orissl_cvm.utils.val import val
from orissl_cvm.utils.tools import save_checkpoint, input_transform
from orissl_cvm.datasets.cvact_dataset import CVACTDataset, Collator
from orissl_cvm.models.siamese import SAFAvgg16
from orissl_cvm.utils.visualize import visualize_dataloader

from tqdm.auto import trange


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training')

    parser.add_argument('--config_path', type=str, default=join(PACKAGE_ROOT_DIR, os.pardir, 'configs/train.ini'),
                        help='File name (with extension) to an ini file that stores most of the configuration data')
    parser.add_argument('--save_path', type=str, default='./work_dirs',
                        help='Path to save checkpoints to')
    parser.add_argument('--resume_path', type=str, default='',
                        help='Full path and name (with extension) to load checkpoint from, for resuming training.')
    parser.add_argument('--dataset_root_dir', type=str, default='/',
                        help='Root directory of dataset')
    parser.add_argument('--identifier', type=str, default='cvact_vgg16',
                        help='Description of this model, e.g. mapillary_nopanos_vgg16_netvlad')
    parser.add_argument('--nEpochs', type=int, default=30, help='number of epochs to train for')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--save_every_epoch', action='store_true', help='Flag to set a separate checkpoint file for each new epoch')
    parser.add_argument('--threads', type=int, default=6, help='Number of threads for each data loader to use')
    parser.add_argument('--gpu_ids', type=str, default="0", help='Visible gpu ids to use')
    parser.add_argument('--nocuda', action='store_true', help='If true, use CPU only. Else use GPU.')

    # Parse arguments
    opt = parser.parse_args()
    print('===> Terminal arguments')
    print(opt)

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    os.environ["MKL_NUM_THREADS"] = str(opt.threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(opt.threads)
    os.environ["OMP_NUM_THREADS"] = str(opt.threads)

    # Load config file
    configfile = opt.config_path
    assert os.path.isfile(configfile)
    config = configparser.ConfigParser()
    config.read(configfile)
    print('===> Config file')
    print(config)

    # CUDA setting
    cuda = not opt.nocuda
    if cuda and not torch.cuda.is_available():
        raise Exception("No GPU found, please run with --nocuda")

    device = torch.device("cuda" if cuda else "cpu")
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())

    # Random seeds
    random.seed(int(config['train']['seed']))
    np.random.seed(int(config['train']['seed']))
    torch.manual_seed(int(config['train']['seed']))
    if cuda:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(int(config['train']['seed']))

    # Model: with resuming or not
    print('===> Building model')

    if opt.resume_path: # if already started training earlier and continuing
        if isfile(opt.resume_path):
            print("=> loading checkpoint '{}'".format(opt.resume_path))
            checkpoint = torch.load(opt.resume_path, map_location=lambda storage, loc: storage)

            model = SAFAvgg16(config['global_params'])

            model.load_state_dict(checkpoint['state_dict'])
            opt.start_epoch = checkpoint['epoch']

            print("=> loaded checkpoint '{}'".format(opt.resume_path, ))
        else:
            raise FileNotFoundError("=> no checkpoint found at '{}'".format(opt.resume_path))
    else: # if not, assume fresh training instance and will initially generate cluster centroids
        print('===> Loading model')
        model = SAFAvgg16(config['global_params'])

    desc_dim = model.desc_dim
    print("===> Model")
    print(model)

    # If DataParallel
    isParallel = False
    if int(config['global_params']['nGPU']) > 1 and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        isParallel = True

    # Optimizer and scheduler
    optimizer = None
    scheduler = None

    if config['train']['optim'] == 'ADAM':
        optimizer = optim.Adam(filter(lambda par: par.requires_grad,
                                      model.parameters()), lr=float(config['train']['lr']))  # , betas=(0,0.9))
    elif config['train']['optim'] == 'SGD':
        optimizer = optim.SGD(filter(lambda par: par.requires_grad,
                                     model.parameters()), lr=float(config['train']['lr']),
                              momentum=float(config['train']['momentum']),
                              weight_decay=float(config['train']['weightDecay']))

        # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(config['train']['lrstep']),
        #                                       gamma=float(config['train']['lrgamma']))
    else:
        raise ValueError('Unknown optimizer: ' + config['train']['optim'])
    
    model = model.to(device)

    if opt.resume_path:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # Loss
    criterion = nn.TripletMarginLoss(margin=float(config['train']['margin']) ** 0.5, p=2, reduction='sum').to(device)
    
    # Dataset and dataloader
    print('===> Loading dataset(s)')
    train_dataset = CVACTDataset(opt.dataset_root_dir, 
                                 mode='train', 
                                 transform=input_transform(),
                                 threads=opt.threads, 
                                 margin=float(config['train']['margin']))
    print(f'Full num of images in training set: {train_dataset.qImages.shape[0]}')
    print(f'Num of queries in training set: {len(train_dataset)}')

    collator = Collator(train_dataset.qpn_matrix)
    training_data_loader = DataLoader(dataset=train_dataset, 
                                      num_workers=opt.threads,
                                      batch_size=int(config['train']['batchsize']), 
                                      shuffle=True,
                                      collate_fn=collator, 
                                      pin_memory=cuda)

    # NOTE visualize batches for debug
    # visualize_dataloader(training_data_loader)

    # validation_dataset = CVACTDataset(opt.dataset_root_dir, 
    #                                   mode='val', 
    #                                   transform=input_transform(),
    #                                 #   posDistThr=25)
    #                                   threads=opt.threads,
    #                                   margin=float(config['train']['margin']))
    validation_dataset = train_dataset

    print('===> Training query set:', len(train_dataset.qIdx))
    print('===> Evaluating on val set, query count:', len(validation_dataset.qIdx))

    # SummaryWriter, and create logdir
    writer = SummaryWriter(
        log_dir=join(opt.save_path, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + opt.identifier))

    logdir = writer.file_writer.get_logdir()
    opt.save_file_path = join(logdir, 'checkpoints')
    makedirs(opt.save_file_path)

    not_improved = 0
    best_score = 0
    if opt.resume_path:
        not_improved = checkpoint['not_improved']
        best_score = checkpoint['best_score']

    #
    # Training
    #
    print('===> Training model')

    # TODO: load from a model. delete it later
    # model.load_state_dict(torch.load("work_dirs/Apr04_21-49-08_cvact_resnet50/checkpoints/a_temp_for_debug.pth"))

    for epoch in trange(opt.start_epoch + 1, opt.nEpochs + 1, desc='Epoch number'.rjust(15), position=0):

        # train_epoch(train_dataset, training_data_loader, model, optimizer, criterion, encoder_dim, device, epoch, opt, config, writer)

        # TODO: delete it later
        # torch.save(model.state_dict(), join(opt.save_file_path, "a_temp_for_debug.pth"))

        if scheduler is not None:
            scheduler.step(epoch)

        if (epoch % int(config['train']['evalevery'])) == 0:
            recalls = val(validation_dataset, model, desc_dim, device, opt, config, writer, epoch,
                          write_tboard=True, pbar_position=1)
            is_best = recalls[5] > best_score
            if is_best:
                not_improved = 0
                best_score = recalls[5]
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
            }, opt, is_best)

            if int(config['train']['patience']) > 0 and not_improved > (int(config['train']['patience']) / int(config['train']['evalevery'])):
                print('Performance did not improve for', config['train']['patience'], 'epochs. Stopping.')
                break

    print("=> Best Recall@5: {:.4f}".format(best_score), flush=True)
    writer.close()

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs

    print('Done')