import sys
from os.path import join
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from sklearn.neighbors import NearestNeighbors
import torch.utils.data as data


class BaseCVMDataset(Dataset):
    def __init__(self, root_dir, mode, transform, logger, 
                 pos_thr=5, neg_thr=5, positive_sampling=False, version='mini'):
        # initializing
        assert mode in ['train', 'val']
        self.root_dir = root_dir
        self.q_imgnames = []
        self.q_idx = []
        self.p_idx = []
        self.nn_idx = []
        self.all_pos_indices = []
        # hyper-parameters
        self.pos_thr = pos_thr
        self.neg_thr = neg_thr
        # flags
        self.mode = mode
        self.version = version
        # other
        self.transform = transform
        self.postive_sampling = positive_sampling
        self.logger = logger

        # # load data
        # q_keys, q_utm = self.read_info()
        # assert(len(q_keys) != 0 and len(q_utm) != 0)
        # self.q_imgnames.extend(q_keys)


    def read_info(self):
        raise NotImplementedError


    def construct_tree(self, q_utm):
        # find positive images for training
        neigh = NearestNeighbors(algorithm='kd_tree')
        self.logger.info(f'Construct neighbor searches for {self.mode} set: {neigh.algorithm}')
        neigh.fit(q_utm)
        pos_distances, pos_indices = neigh.radius_neighbors(q_utm, self.pos_thr)
        self.logger.info(f'Finding positive neighbors for {q_utm.shape[0]} queries')
        self.all_pos_indices.extend(pos_indices)

        if self.mode == 'train':
            self.logger.info(f'Finding non-negative neighbors for {q_utm.shape[0]} queries')
            nnD, nnI = neigh.radius_neighbors(q_utm, self.neg_thr)

        for q_idx in range(len(q_utm)):
            p_idxs = pos_indices[q_idx]
            # the query image has at least one positive
            if len(p_idxs) > 0:
                self.q_idx.append(q_idx)
                self.p_idx.append(p_idxs)
                # in training we have two thresholds, one for finding positives and one 
                # for finding images that we are certain are negatives.
                if self.mode == 'train':
                    n_idxs = nnI[q_idx]
                    self.nn_idx.append(n_idxs)

        if len(self.q_imgnames) == 0:
            self.logger.info("Exiting...")
            self.logger.info("No query images.")
            sys.exit()

        # cast to np.arrays for indexing during training
        self.q_idx = np.asarray(self.q_idx)
        self.q_imgnames = np.asarray(self.q_imgnames)
        self.p_idx = np.asarray(self.p_idx, dtype=object)
        self.nn_idx = np.asarray(self.nn_idx, dtype=object)

        _len = self.q_idx.shape[0]
        self.length = _len
        if self.mode == 'train':
            # create a q-p-n look up matrix
            qpn_matrix = np.zeros((_len, _len))
            for i in range(_len):
                q = self.q_idx[i]
                qpn_matrix[q, self.nn_idx[q].astype(int)] = -1
                qpn_matrix[q, self.p_idx[q].astype(int)] = 1
            self.qpn_matrix = qpn_matrix

            # calculate weights for positive sampling
            if self.positive_sampling:
                self.calcSamplingWeights()
            else:
                self.weights = np.ones(len(self.q_idx)) / float(len(self.q_idx))


    def calcSamplingWeights(self):
        '''TODO possibly give weights to queries in the future'''
        # length of query
        N = len(self.q_idx)
        # initialize weights
        self.weights = np.ones(N)
        # weight higher if ...
        pass
        # print weight information
        pass


    def __getitem__(self, q_idx):
        key = self.q_imgnames[q_idx]
        key_gr, key_sa = key['img_gr'], key['img_sa']
        try:
            input_data = self.transform(Image.open(join(self.gr_path, key_gr)), 
                                        Image.open(join(self.sa_path, key_sa)))
        except (FileNotFoundError, OSError):
            input_data = None
        return input_data, key_gr, key_sa, q_idx


    def __len__(self):
        return self.length


    @staticmethod
    def collate_fn(batch):
        input_data, key_gr, key_sa, q_idx = zip(*batch)
        if None in input_data: return None
        input_data = [data.dataloader.default_collate(x) for x in zip(*input_data)]
        meta = {
            'indices': list(q_idx),
            'keys_gr': list(key_gr),
            'keys_sa': list(key_sa)
        }
        return input_data, meta


class CVACTDataset(BaseCVMDataset):
    def __init__(self, root_dir, mode, transform, logger, pos_thr=5, neg_thr=5, positive_sampling=False, version='mini'):
        super().__init__(root_dir, mode, transform, logger, pos_thr, neg_thr, positive_sampling, version)

        self.gr_path = join(self.root_dir, 'streetview')
        self.sa_path = join(self.root_dir, 'polarmap')

    def read_info(self):
        info_path = join(sys.path[0], f'assets/CVACT_infos_{self.version}')
        assert self.mode in ['train', 'val']
        key_list = []
        utm_list = []
        # load the training, validation, and test set
        with open(join(info_path, f'{self.mode}list.txt'), 'r') as fh:
            for line in fh.readlines():
                # remove linebreak which is the last character of the string
                content = line[:-1].split(' ')
                key_list.append({'img_gr': content[0].split('/')[-1],
                                 'img_sa': content[1].split('/')[-1]})
                utm_list.append(np.array([content[2], content[3]]).astype(np.float64))

        return key_list, np.asarray(utm_list)


