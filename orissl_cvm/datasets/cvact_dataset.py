import sys
from os.path import join
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.utils.data as data
from sklearn.neighbors import NearestNeighbors
import itertools
import math
import random
from tqdm import tqdm


class ImagesFromList(Dataset): #! deprecated
    def __init__(self, images, transform):
        self.images = np.asarray(images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        try:
            img = [Image.open(im) for im in self.images[idx].split(",")]
        except:
            img = [Image.open(self.images[0])]
        img = [self.transform(im) for im in img]

        if len(img) == 1:
            img = img[0]

        return img, idx


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
            img_gr = [Image.open(join(self.gr_path, im)) for im in self.images[idx]['gr_img']]
            img_sa = [Image.open(join(self.sa_path, im)) for im in self.images[idx]['sa_img']]
        except:
            return None
        img_gr = [self.transform(im) for im in img_gr]
        img_sa = [self.transform(im) for im in img_sa]

        if len(img_gr) == 1 or len(img_sa) == 1:
            img_gr = img_gr[0]
            img_sa = img_sa[0]

        return (img_gr, img_sa), idx

    @staticmethod
    def collate_fn(batch):
        return data.dataloader.default_collate(batch) if None not in batch else None


class CVACTDataset(Dataset):
    def __init__(self, root_dir, mode='train', transform=None, posDistThr=15, 
                 negDistThr=100, positive_sampling=True, threads=6, margin=0.1):

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
        self.margin = margin
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr

        # flags
        self.mode = mode

        # other
        self.transform = transform

        # load data # TODO a better way?
        data_info_path = join(sys.path[0], 'assets/CVACT_infos_tiny')

        # when GPS / UTM is available
        if self.mode in ['train', 'val']:
            keysQ, utmQ = self.read_info(data_info_path, self.mode)
            assert(len(keysQ) != 0 and len(utmQ) != 0)

            # train_keys, train_utms = self.read_info(data_info_path, 'train')
            # val_keys, val_utms = self.read_info(data_info_path, 'val')
            # keys = train_keys + val_keys
            # utms = np.concatenate((train_utms, val_utms), axis=0)

            self.qImages.extend(keysQ)
            self.qEndPosList.append(len(keysQ))

            # find positive images for training
            neigh = NearestNeighbors(algorithm='kd_tree')
            print(f'Construct neighbor searches: {neigh.algorithm}')
            neigh.fit(utmQ)
            pos_distances, pos_indices = neigh.radius_neighbors(utmQ, self.posDistThr)
            print(f'Finding positive neighbors for {utmQ.shape[0]} queries')
            self.all_pos_indices.extend(pos_indices)

            if self.mode == 'train':
                print(f'Finding non-negative neighbors for {utmQ.shape[0]} queries')
                nnD, nnI = neigh.radius_neighbors(utmQ, self.negDistThr)

            for q_idx in range(len(keysQ)):
                p_idxs = pos_indices[q_idx]

                # the query image has at least one positive
                if len(p_idxs) > 0:
                    self.qIdx.append(q_idx)
                    self.pIdx.append(p_idxs)

                    # in training we have two thresholds, one for finding positives and one for finding images
                    # that we are certain are negatives.
                    if self.mode == 'train':
                        n_idxs = nnI[q_idx]
                        self.nonNegIdx.append(n_idxs)

        # when GPS / UTM / pano info is not available
        elif self.mode in ['test']:
            # load images for subtask
            keysQ, utmQ = self.read_info(data_info_path, 'test')
            assert(len(keysQ) != 0 and len(utmQ) != 0)

            self.qImages.extend(keysQ)
            # add query index
            self.qIdx.extend(list(range(len(keysQ))))

        if len(self.qImages) == 0:
            print("Exiting...")
            print("No query images.")
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
        self.threads = threads

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
                key_list.append({'gr_img': content[0].split('/')[-1],
                                 'sa_img': content[1].split('/')[-1]})
                utm_list.append(np.array([content[2], content[3]]).astype(np.float64))

        return key_list, np.asarray(utm_list)

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

        # load images into triplet list
        try:
            query_gr = self.transform(Image.open(join(self.gr_path, key['gr_img'])))
            query_sa = self.transform(Image.open(join(self.sa_path, key['sa_img'])))
            query = query_gr, query_sa

        except:
            # NOTE: errors met till now: 
            # FileNotFoundError, OSError: image file is truncated
            # https://stackoverflow.com/a/23575424 could solve it but we choose not to use it
            return None

        return query, key, qidx

    def __len__(self):
        return self.length


class Collator(object):
    def __init__(self, qpn_matrix) -> None:
        self.qpn_matrix = qpn_matrix

    def __call__(self, batch):
        if None in batch:
            return None

        query, key, qidx = zip(*batch)

        query_gr = data.dataloader.default_collate([q[0] for q in query])
        query_sa = data.dataloader.default_collate([q[1] for q in query])
        query = query_gr, query_sa

        qidx, key = list(qidx), list(key)
        batch_qpn_mat = self.qpn_matrix[qidx, :][:, qidx]

        meta = {
            'indices': qidx,
            'keys': key,
            'qpn_mat': batch_qpn_mat
        }

        return query, meta