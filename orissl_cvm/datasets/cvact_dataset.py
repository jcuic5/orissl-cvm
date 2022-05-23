import sys
from os.path import join
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.utils.data as data
from sklearn.neighbors import NearestNeighbors
import random


class ImagesFromList(Dataset):
    def __init__(self, root_dir, images, transform, category='ground'):
        self.root_dir = root_dir
        if category == 'ground':
            self.path = join(self.root_dir, 'streetview')
            self.key = 'img_gr'
        elif category == 'satellite':
            self.path = join(self.root_dir, 'polarmap')
            self.key = 'img_sa'
        else:
            raise NotImplementedError

        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = self.transform(Image.open(join(self.path, self.images[idx][self.key])))
        except:
            return None
        return img, idx

    @staticmethod
    def collate_fn(batch):
        return data.dataloader.default_collate(batch) if None not in batch else None


class ImagePairsFromList(Dataset):
    def __init__(self, root_dir, images, transform):
        self.root_dir = root_dir
        self.gr_path = join(self.root_dir, 'streetview')
        self.sa_path = join(self.root_dir, 'polarmap')

        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img_gr = self.transform(Image.open(join(self.gr_path, self.images[idx]['img_gr'])))
            img_sa = self.transform(Image.open(join(self.sa_path, self.images[idx]['img_sa'])))
        except:
            return None
        return img_gr, img_sa, idx

    @staticmethod
    def collate_fn(batch):
        return data.dataloader.default_collate(batch) if None not in batch else None


class CVACTDataset(Dataset):
    def __init__(self, root_dir, mode, transform, logger, posDistThr=5, 
                 negDistThr=5, positive_sampling=False, mini_scale=None, 
                 version='mini'):

        # initializing
        assert mode in ('train', 'val', 'test')
        self.root_dir = root_dir
        self.gr_path = join(self.root_dir, 'streetview')
        self.sa_path = join(self.root_dir, 'polarmap')
        self.qImages = []
        self.qIdx = []
        self.pIdx = []
        self.nonNegIdx = []
        self.qEndPosList = []
        self.all_pos_indices = []
        # hyper-parameters
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        # flags
        self.mode = mode
        # other
        self.transform = transform
        self.logger = logger

        # load data
        data_info_path = join(sys.path[0], f'assets/CVACT_infos_{version}')
        if self.mode in ['train', 'val']:
            keysQ, utmQ = self.read_info(data_info_path, self.mode)
            assert(len(keysQ) != 0 and len(utmQ) != 0)
            if mini_scale is not None and mini_scale != 1:
                rand = np.random.choice(len(keysQ), int(len(keysQ)*mini_scale), replace=False)
                keysQ = [keysQ[i] for i in rand]
                utmQ = utmQ[rand, :]

            self.qImages.extend(keysQ)
            self.qEndPosList.append(len(keysQ))

            # find positive images for training
            neigh = NearestNeighbors(algorithm='kd_tree')
            self.logger.info(f'Construct neighbor searches for {self.mode} set: {neigh.algorithm}')
            neigh.fit(utmQ)
            pos_distances, pos_indices = neigh.radius_neighbors(utmQ, self.posDistThr)
            self.logger.info(f'Finding positive neighbors for {utmQ.shape[0]} queries')
            self.all_pos_indices.extend(pos_indices)

            if self.mode == 'train':
                self.logger.info(f'Finding non-negative neighbors for {utmQ.shape[0]} queries')
                nnD, nnI = neigh.radius_neighbors(utmQ, self.negDistThr)

            for q_idx in range(len(keysQ)):
                p_idxs = pos_indices[q_idx]

                # the query image has at least one positive
                if len(p_idxs) > 0:
                    self.qIdx.append(q_idx)
                    self.pIdx.append(p_idxs)

                    # in training we have two thresholds, one for finding positives and one 
                    # for finding images that we are certain are negatives.
                    if self.mode == 'train':
                        n_idxs = nnI[q_idx]
                        self.nonNegIdx.append(n_idxs)

        elif self.mode in ['test']:
            # load images for subtask
            keysQ, utmQ = self.read_info(data_info_path, 'test')
            assert(len(keysQ) != 0 and len(utmQ) != 0)
            if mini_scale is not None or mini_scale != 1:
                rand = np.random.choice(len(keysQ), int(len(keysQ)*mini_scale), replace=False)
                keysQ = [keysQ[i] for i in rand]
                utmQ = utmQ[rand, :]

            self.qImages.extend(keysQ)
            # add query index
            self.qIdx.extend(list(range(len(keysQ))))

        if len(self.qImages) == 0:
            self.logger.info("Exiting...")
            self.logger.info("No query images.")
            sys.exit()

        # cast to np.arrays for indexing during training
        self.qIdx = np.asarray(self.qIdx)
        self.qImages = np.asarray(self.qImages)
        self.pIdx = np.asarray(self.pIdx, dtype=object)
        self.nonNegIdx = np.asarray(self.nonNegIdx, dtype=object)

        _len = self.qIdx.shape[0]
        self.length = _len
        if mode == 'train':
            # create a q-p-n look up matrix
            qpn_matrix = np.zeros((_len, _len))
            for i in range(_len):
                q = self.qIdx[i]
                qpn_matrix[q, self.nonNegIdx[q].astype(int)] = -1
                qpn_matrix[q, self.pIdx[q].astype(int)] = 1
            self.qpn_matrix = qpn_matrix

        # decide device type ( important for triplet mining )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        if mode == 'train':
            # calculate weights for positive sampling
            if positive_sampling:
                self.__calcSamplingWeights__()
            else:
                self.weights = np.ones(len(self.qIdx)) / float(len(self.qIdx))

    def read_info(self, info_path, mode):
        assert mode in ['train', 'val', 'test']
        key_list = []
        utm_list = []
        # load the training, validation, and test set
        with open(join(info_path, f'{mode}list.txt'), 'r') as fh:
            for line in fh.readlines():
                # remove linebreak which is the last character of the string
                content = line[:-1].split(' ')
                key_list.append({'img_gr': content[0].split('/')[-1],
                                 'img_sa': content[1].split('/')[-1]})
                utm_list.append(np.array([content[2], content[3]]).astype(np.float64))

        return key_list, np.asarray(utm_list)

    # TODO possibly give weights to queries in the future
    def __calcSamplingWeights__(self):
        # length of query
        N = len(self.qIdx)
        # initialize weights
        self.weights = np.ones(N)
        # weight higher if ...
        pass
        # print weight information
        pass

    def __getitem__(self, qidx):
        key = self.qImages[qidx]
        key_gr, key_sa = key['img_gr'], key['img_sa']
        try:
            input_data = self.transform(Image.open(join(self.gr_path, key_gr)), 
                                        Image.open(join(self.sa_path, key_sa)))
        except (FileNotFoundError, OSError):
            input_data = None
        return input_data, key_gr, key_sa, qidx

    def __len__(self):
        return self.length

    @staticmethod
    def collate_fn(batch):
        input_data, key_gr, key_sa, qidx = zip(*batch)
        if None in input_data: return None
        input_data = [data.dataloader.default_collate(x) for x in zip(*input_data)]
        meta = {
            'indices': list(qidx),
            'keys_gr': list(key_gr),
            'keys_sa': list(key_sa)
        }
        return input_data, meta