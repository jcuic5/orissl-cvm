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

Additional functions used during training.
'''


from email.policy import strict
from matplotlib.pyplot import axis
import numpy as np
from scipy.sparse.linalg import eigs
import torch
import shutil
from os.path import join
from PIL import Image
import logging
from easydict import EasyDict
from os.path import join, isfile
from orissl_cvm.models import get_model


def pca(x: np.ndarray, num_pcs=None, subtract_mean=True):
    # translated from MATLAB:
    # - https://github.com/Relja/relja_matlab/blob/master/relja_PCA.m
    # - https://github.com/Relja/netvlad/blob/master/addPCA.m

    # assumes x = nvectors x ndims
    x = x.T  # matlab code is ndims x nvectors, so transpose

    n_points = x.shape[1]
    n_dims = x.shape[0]

    if num_pcs is None:
        num_pcs = n_dims

    print('PCA for {} points of dimension {} to PCA dimension {}'.format(n_points, n_dims, num_pcs))

    if subtract_mean:
        # Subtract mean
        mu = np.mean(x, axis=1)
        x = (x.T - mu).T
    else:
        mu = np.zeros(n_dims)

    assert num_pcs < n_dims

    if n_dims <= n_points:
        do_dual = False
        # x2 = dims * dims
        x2 = np.matmul(x, x.T) / (n_points - 1)
    else:
        do_dual = True
        # x2 = vectors * vectors
        x2 = np.matmul(x.T, x) / (n_points - 1)

    if num_pcs < x2.shape[0]:
        print('Compute {} eigenvectors'.format(num_pcs))
        lams, u = eigs(x2, num_pcs)
    else:
        print('Compute eigenvectors')
        lams, u = np.linalg.eig(x2)

    assert np.all(np.isreal(lams)) and np.all(np.isreal(u))
    lams = np.real(lams)
    u = np.real(u)

    sort_indices = np.argsort(lams)[::-1]
    lams = lams[sort_indices]
    u = u[:, sort_indices]

    if do_dual:
        # U = x * ( U * diag(1./sqrt(max(lams,1e-9))) / sqrt(nPoints-1) );
        diag = np.diag(1. / np.sqrt(np.maximum(lams, 1e-9)))
        utimesdiag = np.matmul(u, diag)
        u = np.matmul(x, utimesdiag / np.sqrt(n_points - 1))

    return u, lams, mu


def humanbytes(B):
    """Return the given bytes as a human friendly KB, MB, GB, or TB string"""
    B = float(B)
    KB = float(1024)
    MB = float(KB ** 2)  # 1,048,576
    GB = float(KB ** 3)  # 1,073,741,824
    TB = float(KB ** 4)  # 1,099,511,627,776

    if B < KB:
        return '{0} {1}'.format(B,'Bytes' if 0 == B > 1 else 'Byte')
    elif KB <= B < MB:
        return '{0:.2f} KB'.format(B/KB)
    elif MB <= B < GB:
        return '{0:.2f} MB'.format(B/MB)
    elif GB <= B < TB:
        return '{0:.2f} GB'.format(B/GB)
    elif TB <= B:
        return '{0:.2f} TB'.format(B/TB)


def save_checkpoint(state, config, is_best_sofar, filename='checkpoint.pth.tar'):
    if config.train.save_every_epoch:
        model_out_path = join(config.train.save_file_path, 'checkpoint_epoch' + str(state['epoch']) + '.pth.tar')
    else:
        model_out_path = join(config.train.save_file_path, filename)
    torch.save(state, model_out_path)
    if is_best_sofar:
        shutil.copyfile(model_out_path, join(config.train.save_file_path, 'model_best.pth.tar'))


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    # NOTE code from OpenPCDet
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def log_config_to_file(cfg, pre='cfg', logger=None):
    # NOTE code from OpenPCDet
    for key, val in cfg.items():
        if isinstance(cfg[key], EasyDict):
            logger.info('\n%s.%s = edict()' % (pre, key))
            log_config_to_file(cfg[key], pre=pre + '.' + key, logger=logger)
            continue
        logger.info('%s.%s: %s' % (pre, key, val))


def get_model_with_ckpt(cfg, logger):
    model, checkpoint = None, None

    if cfg.train.resume_path and cfg.train.load_path:
        raise RuntimeError("===> Resume path and load path are both indicated!")

    elif cfg.train.load_path:
        if cfg.train.load_sep_branch:
            load_path_gr, load_path_sa = cfg.train.load_path.split(',')
            if isfile(load_path_gr) and isfile(load_path_sa):
                logger.info("===> loading branch weights separately '{}'".format(cfg.train.load_path))
                encoder_gr = torch.load(load_path_gr, map_location=lambda storage, loc: storage)
                encoder_sa = torch.load(load_path_sa, map_location=lambda storage, loc: storage)
                model = get_model(cfg.model)
                # NOTE For debug, check if params are loaded
                # p1 = next(iter(model.nn_model_gr.backbone.parameters()))
                # p2 = encoder_gr['state_dict']['backbone.0.weight']
                model.nn_model_gr.load_state_dict({k : v for k, v in encoder_gr['state_dict'].items() if k.startswith('backbone.') or k.startswith('pool.')}, strict=True)
                model.nn_model_sa.load_state_dict({k : v for k, v in encoder_sa['state_dict'].items() if k.startswith('backbone.') or k.startswith('pool.')}, strict=True)
                logger.info("===> loaded branch weights separately '{}'".format(cfg.train.load_path))
            else:
                raise FileNotFoundError("===> no checkpoint found at '{}'".format(cfg.train.load_path))
        else:
            if isfile(cfg.train.load_path):                    
                logger.info("===> loading model weights '{}'".format(cfg.train.load_path))
                checkpoint = torch.load(cfg.train.load_path, map_location=lambda storage, loc: storage)
                model = get_model(cfg.model)
                if cfg.train.load_only_backbone_pool:
                    model.load_state_dict({k : v for k, v in checkpoint['state_dict'].items() if k.startswith('backbone.') or k.startswith('pool.')}, strict=True)
                else:
                    model.load_state_dict(checkpoint['state_dict'], strict=True)
                logger.info("===> loaded model weights '{}'".format(cfg.train.load_path))
            else:
                raise FileNotFoundError("===> no checkpoint found at '{}'".format(cfg.train.load_path))

    elif cfg.train.resume_path: # if already started training earlier and continuing
        if isfile(cfg.train.resume_path):
            logger.info("===> loading checkpoint '{}'".format(cfg.train.resume_path))
            checkpoint = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage)
            model = get_model(cfg.model)
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            cfg.train.start_epoch = checkpoint['epoch']
            logger.info("===> loaded checkpoint '{}'".format(cfg.train.resume_path))
        else:
            raise FileNotFoundError("===> no checkpoint found at '{}'".format(cfg.train.resume_path))

    else: # if not, assume fresh training instance and will initially generate cluster centroids
        model = get_model(cfg.model)

    return model, checkpoint


def backward_hook(module, grad_in, grad_out, grad_list):
    grad_list.append(grad_out[0].detach())


def forward_hook(module, input, output, fmp_list):
    fmp_list.append(output)


def gen_cam(fmp, grad):
    """ 依据梯度和特征图，生成cam
    :param feature_map: np.array， in [C, H, W]
    :param grads: np.array， in [C, H, W]
    :return: np.array, [H, W]
    """
    B, C, H, W = fmp.shape
    cam = np.zeros((B, H, W), dtype=np.float32)  # (B, H, W)
    weights = np.mean(grad, axis=(2, 3)) # (B, C)
    # weights[...] = 1
    for b in range(B):
        for c in range(C):
            cam[b] += weights[b, c] * fmp[b, c, :, :]
    # cam = np.maximum(cam, 0) # relu
    # cam -= np.min(cam, axis=0)
    # cam /= np.max(cam, axis=0)
    return cam


def show_cam_on_image(img, mask, out_dir):
    heatmap = cv2.applyColorMap(np.uint8(255*mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)

    path_cam_img = os.path.join(out_dir, "cam.jpg")
    path_raw_img = os.path.join(out_dir, "raw.jpg")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    cv2.imwrite(path_cam_img, np.uint8(255 * cam))
    cv2.imwrite(path_raw_img, np.uint8(255 * img))