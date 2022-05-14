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
import imp
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
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
import copy
import matplotlib.cm as mpl_color_map


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
    model, ckpt = None, None

    if cfg.train.resume_path and cfg.train.load_path:
        raise RuntimeError("===> Resume path and load path are both indicated!")
    elif cfg.train.load_path:
        if cfg.train.load_sep_branch:
            load_path_gr, load_path_sa = cfg.train.load_path.split(',')
            if isfile(load_path_gr) and isfile(load_path_sa):
                logger.info("===> loading branch weights separately '{}'".format(cfg.train.load_path))
                ckpt_gr = torch.load(load_path_gr, map_location=lambda storage, loc: storage)
                ckpt_sa = torch.load(load_path_sa, map_location=lambda storage, loc: storage)
                model = get_model(cfg.model)
                # NOTE For debug, check if params are loaded
                # p1 = next(iter(model.model_gr.backbone.parameters()))
                # p2 = model_gr['state_dict']['backbone.0.weight']
                model.features_gr.load_state_dict({k[9:] : v for k, v in ckpt_gr['state_dict'].items() if k.startswith('features.')}, strict=True)
                model.features_sa.load_state_dict({k[9:] : v for k, v in ckpt_sa['state_dict'].items() if k.startswith('features.')}, strict=True)
                logger.info("===> loaded branch weights separately '{}'".format(cfg.train.load_path))
            else:
                raise FileNotFoundError("===> no checkpoint found at '{}'".format(cfg.train.load_path))
        else:
            if isfile(cfg.train.load_path):                    
                logger.info("===> loading model weights '{}'".format(cfg.train.load_path))
                ckpt = torch.load(cfg.train.load_path, map_location=lambda storage, loc: storage)
                model = get_model(cfg.model)
                if cfg.train.load_only_backbone:
                    model.load_state_dict({k[9:] : v for k, v in ckpt['state_dict'].items() if k.startswith('features.')}, strict=True)
                else:
                    model.load_state_dict(ckpt['state_dict'], strict=True)
                logger.info("===> loaded model weights '{}'".format(cfg.train.load_path))
            else:
                raise FileNotFoundError("===> no checkpoint found at '{}'".format(cfg.train.load_path))

    elif cfg.train.resume_path: # if already started training earlier and continuing
        if isfile(cfg.train.resume_path):
            logger.info("===> loading checkpoint '{}'".format(cfg.train.resume_path))
            ckpt = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage)
            model = get_model(cfg.model)
            model.load_state_dict(ckpt['state_dict'], strict=True)
            cfg.train.start_epoch = ckpt['epoch']
            logger.info("===> loaded checkpoint '{}'".format(cfg.train.resume_path))
        else:
            raise FileNotFoundError("===> no checkpoint found at '{}'".format(cfg.train.resume_path))
    else: # if not, assume fresh training instance and will initially generate cluster centroids
        model = get_model(cfg.model)

    return model, ckpt


def backward_hook(module, grad_in, grad_out, grad_list):
    grad_list.append(grad_out[0].detach())


def forward_hook(module, input, output, fmp_list):
    fmp_list.append(output)


def gen_cam(fmp, grad):
    """ 依据梯度和特征图，生成cam
    :param feature_map: np.array， in [B, C, H, W]
    :param grads: np.array， in [B, C, H, W]
    :return: np.array, [B, H, W]
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


def show_cam_on_image(img, heatmap):
    """Apply heatmap on image
    """
    B, C, H, W = img.shape
    im = img.clone().detach()
    color_map = mpl_color_map.get_cmap('viridis')
    # heatmap = resize(heatmap, (H, W), InterpolationMode.NEAREST).unsqueeze(dim=1).to(img.device)
    heatmap = resize(heatmap, (H, W), InterpolationMode.BILINEAR).unsqueeze(dim=1)
    for i in range(B):
        hmap = heatmap[i].squeeze(0).numpy()
        hmap -= np.min(hmap)
        hmap /= np.max(hmap)
        # cmap = color_map(hmap)
        # cmap = torch.tensor(cmap).to(img.device).permute(2, 0, 1)
        # img[i] += cmap[:3] * cmap[3]
        im[i] += torch.tensor(hmap).to(im.device).unsqueeze(dim=0) * 15
    return im


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on image
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image