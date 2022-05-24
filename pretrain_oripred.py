from __future__ import print_function

import argparse
import yaml
from easydict import EasyDict as edict  
import os
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
from torch.utils.data import DataLoader

from clsslcvm import PACKAGE_ROOT_DIR
from clsslcvm.models import get_model
from clsslcvm.tools.pretrain_oripred_epoch import train_epoch
from clsslcvm.tools.val_oripred import val
from clsslcvm.tools import save_checkpoint, create_logger, log_config_to_file, get_model_with_ckpt
from clsslcvm.augmentations import input_transform, OriLabelPairTransform
from clsslcvm.datasets.cvact_dataset import CVACTDataset
from clsslcvm.tools.visualize import visualize_dataloader

from tqdm.auto import trange


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='training')
    parser.add_argument('--config_path', type=str, default=join(PACKAGE_ROOT_DIR, os.pardir, 'configs/pretrain_oripred.yaml'),
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
    random.seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)
    if cuda:
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(cfg.train.seed)

    # Model: with resuming or not
    logger.info('===> Building model')
    model, checkpoint = get_model_with_ckpt(cfg, logger)
    model = model.to(device)
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
    # criterion = nn.CrossEntropyLoss().to(device)
    criterion = nn.MSELoss().to(device)
    
    # Dataset and dataloader
    logger.info('===> Loading dataset(s)')
    train_dataset = CVACTDataset(cfg.dataset.dataset_root_dir,
                                 mode='train', 
                                 transform=OriLabelPairTransform((cfg.model.img_size_h, cfg.model.img_size_w)),
                                 logger=logger,
                                 version=cfg.dataset.dataset_version)
    logger.info(f'Full num of image pairs in training set: {train_dataset.q_imgnames.shape[0]}')
    logger.info(f'Num of queries in training set: {len(train_dataset)}')

    training_data_loader = DataLoader(dataset=train_dataset, 
        num_workers=cfg.dataset.n_workers,
        batch_size=cfg.train.batch_size, 
        shuffle=cfg.dataset.train_loader_shuffle,
        collate_fn = train_dataset.collate_fn, 
        pin_memory=cuda
    )

    # NOTE visualize batches for debug
    # visualize_dataloader(training_data_loader)

    if cfg.train.train_as_val is True:
        # NOTE for debug, use train set it self to validate
        val_dataset = train_dataset
    else:
        val_dataset = CVACTDataset(cfg.dataset.dataset_root_dir, 
                                    mode='val', 
                                    transform=OriLabelPairTransform((cfg.model.img_size_h, cfg.model.img_size_w)), # NOTE because we're not really using this dataset
                                    logger=logger,
                                    version=cfg.dataset.dataset_version)
    logger.info(f'Full num of image pairs in validation set: {val_dataset.q_imgnames.shape[0]}')
    logger.info(f'Num of queries in validation set: {len(val_dataset)}')
    val_data_loader = DataLoader(dataset=val_dataset, 
        num_workers=cfg.dataset.n_workers,
        batch_size=cfg.train.batch_size, 
        shuffle=False,
        collate_fn=val_dataset.collate_fn, 
        pin_memory=cuda
    )

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
    for epoch in trange(cfg.train.start_epoch + 1, cfg.train.n_epochs + 1, desc='Epoch number'.rjust(15), position=0):
        train_epoch(train_dataset, training_data_loader, model, optimizer, scheduler, criterion, device, epoch, cfg, writer)
        
        if (epoch % cfg.train.eval_every) == 0:
            score = val(val_data_loader, model, device, cfg, writer, epoch, write_tboard=True, pbar_position=1)
            is_best = score > best_score
            if is_best:
                not_improved = 0
                best_score = score
            else:
                not_improved += 1

            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'score': score,
                'best_score': best_score,
                'not_improved': not_improved,
                'optimizer': optimizer.state_dict(),
                'parallel': isParallel,
            }, cfg, is_best)

            if cfg.train.patience > 0 and not_improved > (cfg.train.patience / cfg.train.eval_every):
                logger.info(f"Performance did not improve for {cfg.train.patience} epochs. Stopping.")
                break

    logger.info("=> Best accuracy: {:.4f}".format(best_score))
    writer.close()

    torch.cuda.empty_cache()  # garbage clean GPU memory, a bug can occur when Pytorch doesn't automatically clear the
    # memory after runs

    logger.info('Done')