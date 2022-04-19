'''
Copyright (c) Facebook, Inc. and its affiliates.

MIT License

Copyright (c) 2020 mapillary

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

Modified by Jianfeng Cui
'''
import sys
from os.path import join
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torch.utils.data as data
from sklearn.neighbors import NearestNeighbors
import random


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
            img_gr = self.transform(Image.open(join(self.gr_path, self.images[idx]['gr_img'])))
            img_sa = self.transform(Image.open(join(self.sa_path, self.images[idx]['sa_img'])))
        except:
            return None
        return img_gr, img_sa, idx

    @staticmethod
    def collate_fn(batch):
        return data.dataloader.default_collate(batch) if None not in batch else None


class CVACTDataset(Dataset):
    def __init__(self, root_dir, mode, transform, logger, posDistThr=5, 
                 negDistThr=5, positive_sampling=False, mini_scale=None, 
                 task='cvm'):

        # initializing
        assert mode in ('train', 'val', 'test')
        assert task in ('cvm', 'ssl')
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
        self.task = task
        # other
        self.transform = transform
        self.num_dirs = 4
        self.logger = logger

        # load data
        data_info_path = join(sys.path[0], 'assets/CVACT_infos_1')
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
                key_list.append({'gr_img': content[0].split('/')[-1],
                                 'sa_img': content[1].split('/')[-1]})
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

    def random_slide_pano(self, img, label=None):
        H, W = img.shape[-2], img.shape[-1]
        if label == None:
            label = random.randint(0, self.num_dirs)
        slide_w = int(float(1 - label / self.num_dirs) * W)
        # img[..., :5] = 0 # NOTE draw a dividing line for debug
        img = torch.cat([img[..., slide_w:], img[..., :slide_w]], dim=-1)
        # NOTE one-hot encoding, but not needed
        # label = torch.zeros(36, dtype=torch.long).scatter_(dim=0, index=torch.tensor(label), value=1)
        return img, label

    def __getitem__(self, qidx):
        key = self.qImages[qidx]

        if self.task == 'cvm':
            # load images
            try:
                query_gr = self.transform(Image.open(join(self.gr_path, key['gr_img'])))
                query_sa = self.transform(Image.open(join(self.sa_path, key['sa_img'])))
            except:
                # NOTE errors might met: 
                # FileNotFoundError, OSError: image file is truncated
                # https://stackoverflow.com/a/23575424 could solve the truncated issue 
                # but we choose directly not to use the sample
                return None
            return query_gr, query_sa, key, qidx

        elif self.task == 'ssl':
            query_gr, query_sa, label = [], [], []
            # load images
            try:
                gr_img = Image.open(join(self.gr_path, key['gr_img']))
                sa_img = Image.open(join(self.sa_path, key['sa_img']))
            except:
                return None
            for j in range(self.num_dirs):
                for i in range(self.num_dirs):
                    q_gr, l_gr = self.random_slide_pano(self.transform(gr_img), label=i)
                    query_gr.append(q_gr)
                    q_sa, l_sa = self.random_slide_pano(self.transform(sa_img), label=j)
                    query_sa.append(q_sa)
                    label.append(l_gr - l_sa if l_gr - l_sa >= 0 else l_gr - l_sa + self.num_dirs)
            key = [key] * self.num_dirs**2
            qidx = [qidx] * self.num_dirs**2
            return query_gr, query_sa, label, key, qidx

    def __len__(self):
        return self.length

    @staticmethod
    def collate_fn_cvm(batch, qpn_matrix):
        if None in batch:
            return None
        query_gr, query_sa, key, qidx = zip(*batch)

        query_gr = data.dataloader.default_collate(query_gr)
        query_sa = data.dataloader.default_collate(query_sa)
        qidx, key = list(qidx), list(key)
        batch_qpn_mat = qpn_matrix[qidx, :][:, qidx]
        meta = {
            'indices': qidx,
            'keys': key,
            'qpn_mat': batch_qpn_mat
        }
        return query_gr, query_sa, meta

    @staticmethod
    def collate_fn_ssl(batch):
        if None in batch:
            return None
        query_gr, query_sa, label, key, qidx = zip(*batch)
        stack = lambda x : data.dataloader.default_collate(
                        [var for sample in x for var in sample])
        query_gr = stack(query_gr)
        query_sa = stack(query_sa)
        label = stack(label)
        qidx = [var for sample in list(qidx) for var in sample]
        key = [var for sample in list(key) for var in sample]
        meta = {
            'indices': qidx,
            'keys': key
        }
        return query_gr, query_sa, label, meta