from statistics import mode
import numpy as np
from scipy.sparse.linalg import eigs
import torch
import shutil
from os.path import join
from PIL import Image
import logging
from easydict import EasyDict
from os.path import join, isfile
from clsslcvm.models import get_model
from torchvision.transforms.functional import resize
from enum import Enum
import copy
import matplotlib.cm as mpl_color_map
from torch.utils.data.sampler import RandomSampler
from clsslcvm.datasets.vigor_dataset import VIGORDataset, NonOverlapBatchSampler
from clsslcvm.datasets.generic_dataset import ImagesFromList
from torch.utils.data import DataLoader
from clsslcvm.augmentations import input_transform, SimSiamTransform


class InterpolationMode(Enum):
    """Interpolation modes
    """
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    # For PIL compatibility
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


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


def save_checkpoint(state, desc_grd_all, desc_sat_all, cfg, is_best_sofar, filename='checkpoint.pth.tar'):
    if cfg.train.save_every_epoch:
        model_out_path = join(cfg.train.save_file_path, 'checkpoint_epoch' + str(state['epoch']) + '.pth.tar')
        if desc_grd_all is not None and desc_sat_all is not None:
            np.save(join(cfg.train.save_file_path, 'desc_sat_all_epoch' + str(state['epoch']) + '.npy'), desc_sat_all)
            np.save(join(cfg.train.save_file_path, 'desc_grd_all_epoch' + str(state['epoch']) + '.npy'), desc_grd_all)
    else:
        model_out_path = join(cfg.train.save_file_path, filename)
        if desc_grd_all is not None and desc_sat_all is not None:
            np.save(join(cfg.train.save_file_path, 'desc_sat_all.npy'), desc_sat_all)
            np.save(join(cfg.train.save_file_path, 'desc_grd_all.npy'), desc_grd_all)
    torch.save(state, model_out_path)
    if is_best_sofar:
        shutil.copyfile(model_out_path, join(cfg.train.save_file_path, 'model_best.pth.tar'))


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


def load_branch(model, ckpt, branch):
    # rename moco pre-trained keys
    state_dict = ckpt['state_dict']
    for k in list(state_dict.keys()):
        # retain only encoder_q up to before the embedding layer
        if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
            # remove prefix
            state_dict[k[len("module.encoder_q."):]] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    # msg = model.load_state_dict(state_dict, strict=False)
    # assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
    getattr(model, f'features_{branch}').load_state_dict({k[9:] : v for k, v in state_dict.items() if k.startswith('features.')}, strict=True)


def get_model_with_ckpt(cfg, logger):
    model, ckpt = None, None

    if cfg.train.resume_path and cfg.train.load_path:
        raise RuntimeError("resume path and load path are both indicated!")
    elif cfg.train.load_path:
        # contrastive learning pretrained model
        if cfg.train.load_sep_branch:
            # load single branch
            if cfg.train.load_sep_branch_single != '':
                if isfile(cfg.train.load_path):                    
                    logger.info("loading model weights '{}'".format(cfg.train.load_path))
                    ckpt = torch.load(cfg.train.load_path, map_location=lambda storage, loc: storage)
                    model = get_model(cfg.model)
                    if cfg.train.load_sep_branch_single == 'gr':
                        load_branch(model, ckpt, branch='gr')
                    elif cfg.train.load_sep_branch_single == 'sa':
                        load_branch(model, ckpt, branch='sa')
                    logger.info("loaded model weights '{}'".format(cfg.train.load_path))
            # load both branches
            else:
                load_path_gr, load_path_sa = cfg.train.load_path.split(',')
                if isfile(load_path_gr) and isfile(load_path_sa):
                    logger.info("loading branch weights separately '{}'".format(cfg.train.load_path))
                    ckpt_gr = torch.load(load_path_gr, map_location=lambda storage, loc: storage)
                    ckpt_sa = torch.load(load_path_sa, map_location=lambda storage, loc: storage)
                    model = get_model(cfg.model)
                    load_branch(model, ckpt_gr, branch='gr')
                    load_branch(model, ckpt_sa, branch='sa')
                    # NOTE For debug, check if params are loaded
                    # p1 = next(iter(model.model_gr.backbone.parameters()))
                    # p2 = model_gr['state_dict']['backbone.0.weight']
                    logger.info("loaded branch weights separately '{}'".format(cfg.train.load_path))
                else:
                    raise FileNotFoundError("no checkpoint found at '{}'".format(cfg.train.load_path))
        # model itself / contrastive learning pretrained model (shared)
        #! not updated with mocov2 yet
        else:
            if isfile(cfg.train.load_path):                    
                logger.info("loading model weights '{}'".format(cfg.train.load_path))
                ckpt = torch.load(cfg.train.load_path, map_location=lambda storage, loc: storage)
                model = get_model(cfg.model)
                if cfg.train.load_only_backbone:
                    # contrastive learning pretrained model (shared)
                    # model.features_gr.load_state_dict({k[12:] : v for k, v in ckpt['state_dict'].items() if k.startswith('features_gr.')}, strict=True)
                    # model.features_sa.load_state_dict({k[12:] : v for k, v in ckpt['state_dict'].items() if k.startswith('features_sa.')}, strict=True)
                    # contrastive learning downloaded resnet50 model
                    model.features_gr.load_state_dict({k[15:] : v for k, v in ckpt['state_dict'].items() if k.startswith('module.encoder.') and not k.startswith('module.encoder.fc.')}, strict=True)
                    model.features_sa.load_state_dict({k[15:] : v for k, v in ckpt['state_dict'].items() if k.startswith('module.encoder.') and not k.startswith('module.encoder.fc.')}, strict=True)
                # model itself
                else:
                    model.load_state_dict(ckpt['state_dict'], strict=True)
                logger.info("loaded model weights '{}'".format(cfg.train.load_path))
            else:
                raise FileNotFoundError("no checkpoint found at '{}'".format(cfg.train.load_path))

    elif cfg.train.resume_path: # if already started training earlier and continuing
        if isfile(cfg.train.resume_path):
            logger.info("loading checkpoint '{}'".format(cfg.train.resume_path))
            ckpt = torch.load(cfg.train.resume_path, map_location=lambda storage, loc: storage)
            model = get_model(cfg.model)
            model.load_state_dict(ckpt['state_dict'], strict=True)
            cfg.train.start_epoch = ckpt['epoch']
            logger.info("loaded checkpoint '{}'".format(cfg.train.resume_path))
        else:
            raise FileNotFoundError("no checkpoint found at '{}'".format(cfg.train.resume_path))
    else: # if not, assume fresh training instance and will initially generate cluster centroids
        model = get_model(cfg.model)

    return model, ckpt


def get_dataset(cfg, logger):
    if cfg.model.name == 'cvm':
        tf = input_transform
        if cfg.dataset.name == 'cvact':
            raise NotImplementedError
        elif cfg.dataset.name == 'vigor':
            train_set = VIGORDataset(cfg.dataset.dataset_root_dir, 'train', transform=tf, logger=logger, version=cfg.dataset.version)
            train_bsampler = NonOverlapBatchSampler(
                RandomSampler(train_set), 
                train_set.grd_sat_label, 
                batch_size=cfg.train.batch_size, 
                drop_last=False
            )
            train_loader = DataLoader(dataset=train_set, 
                num_workers=cfg.dataset.n_workers, 
                batch_sampler=train_bsampler, 
                pin_memory=True
            )
            val_set = VIGORDataset(cfg.dataset.dataset_root_dir, 'val', transform=tf, logger=logger, version=cfg.dataset.version)
            val_set_queries_grd = ImagesFromList(val_set.grd_list, val_set.grd_size, tf)
            val_set_queries_sat = ImagesFromList(val_set.sat_list, val_set.sat_size, tf)
            opt = {
                'batch_size': cfg.train.batch_size, 
                'shuffle': False, 
                'num_workers': cfg.dataset.n_workers, 
                'pin_memory': True, 
            }
            val_loader_queries_grd = DataLoader(dataset=val_set_queries_grd, **opt)
            val_loader_queries_sat = DataLoader(dataset=val_set_queries_sat, **opt)
        else:
            raise NotImplementedError
        return train_set, train_loader, val_set, val_set_queries_grd, val_set_queries_sat, \
                                                        val_loader_queries_grd, val_loader_queries_sat
    elif cfg.model.name == 'simsiam':
        tf = SimSiamTransform
        if cfg.dataset.name == 'cvact':
            raise NotImplementedError
        elif cfg.dataset.name == 'vigor':
            train_set = VIGORDataset(cfg.dataset.dataset_root_dir, 'train', transform=tf, logger=logger, version=cfg.dataset.version)
            train_set_queries_grd = ImagesFromList(train_set.grd_list, train_set.grd_size, tf)
            train_set_queries_sat = ImagesFromList(train_set.sat_list, train_set.sat_size, tf)
            opt = {
                'num_workers': cfg.dataset.n_workers, 
                'batch_size': cfg.train.batch_size, 
                'shuffle': True, 
                'pin_memory': True,
                'drop_last': True
            }
            train_loader_queries_grd = DataLoader(dataset=train_set_queries_grd, **opt)
            train_loader_queries_sat = DataLoader(dataset=train_set_queries_sat, **opt)        
            return train_set, train_set_queries_grd, train_set_queries_sat, \
                                        train_loader_queries_grd, train_loader_queries_sat
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

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
    cam = np.maximum(cam, 0) # relu
    cam -= np.min(cam, axis=0)
    cam /= np.max(cam, axis=0)
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
    """Apply heatmap on image
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


def register_hook(model, mode='double'):
    '''Register hook on the last layer of backbone and pooling before normalization'''
    if mode == 'double':
        fmp_list_gr, fmp_list_sa = [], []
        grad_list_gr, grad_list_sa = [], []
        fh_gr = lambda module, input, output : forward_hook(module, input, output, fmp_list=fmp_list_gr)
        bh_gr = lambda module, grad_in, grad_out : backward_hook(module, grad_in, grad_out, grad_list=grad_list_gr)
        fh_sa = lambda module, input, output : forward_hook(module, input, output, fmp_list=fmp_list_sa)
        bh_sa = lambda module, grad_in, grad_out : backward_hook(module, grad_in, grad_out, grad_list=grad_list_sa)
        model.features_gr[-1].register_forward_hook(fh_gr)
        model.features_gr[-1].register_full_backward_hook(bh_gr)
        model.features_sa[-1].register_forward_hook(fh_sa)
        model.features_sa[-1].register_full_backward_hook(bh_sa)
        
        d_list = []
        fh_d = lambda module, input, output : forward_hook(module, input, output, fmp_list=d_list)
        model.pool[-2].register_forward_hook(fh_d)

        return fmp_list_gr, fmp_list_sa, grad_list_gr, grad_list_sa, d_list

    elif mode == 'single':
        fmp_list = []
        grad_list = []
        fh = lambda module, input, output : forward_hook(module, input, output, fmp_list=fmp_list)
        bh = lambda module, grad_in, grad_out : backward_hook(module, grad_in, grad_out, grad_list=grad_list)
        model.features[-1].register_forward_hook(fh)
        model.features[-1].register_full_backward_hook(bh)
        
        d_list = []
        fh_d = lambda module, input, output : forward_hook(module, input, output, fmp_list=d_list)
        model.pool[-2].register_forward_hook(fh_d)

        return fmp_list, grad_list, d_list
    
    else:
        return NotImplementedError