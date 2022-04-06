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
    def __init__(self, root_dir, mode='train', nNeg=5, transform=None, posDistThr=15, 
                 negDistThr=100, cached_queries=1000, cached_negatives=1000, 
                 positive_sampling=True, bs=32, threads=6, margin=0.1):

        # initializing
        assert mode in ('train', 'val', 'test')

        self.root_dir = root_dir
        self.gr_path = join(self.root_dir, 'streetview')
        self.sa_path = join(self.root_dir, 'polarmap')

        self.qIdx = []
        self.qImages = []
        self.pIdx = []
        self.nonNegIdx = []
        self.dbImages = []
        self.qEndPosList = []
        self.dbEndPosList = []

        self.all_pos_indices = []

        # hyper-parameters
        self.nNeg = nNeg
        self.margin = margin
        self.posDistThr = posDistThr
        self.negDistThr = negDistThr
        self.cached_queries = cached_queries
        self.cached_negatives = cached_negatives

        # flags
        self.cache = None
        self.mode = mode

        # other
        self.transform = transform

        # load data
        # TODO a better way?
        data_info_path = join(sys.path[0], 'assets/CVACT_infos_mini')
        # get len of images from cities so far for indexing
        _lenQ = len(self.qImages)
        _lenDb = len(self.dbImages)

        # when GPS / UTM is available
        if self.mode in ['train', 'val']:

            keys, utms = self.read_info(data_info_path, self.mode)

            # train_keys, train_utms = self.read_info(data_info_path, 'train')
            # val_keys, val_utms = self.read_info(data_info_path, 'val')
            # keys = train_keys + val_keys
            # utms = np.concatenate((train_utms, val_utms), axis=0)

            assert(len(keys) != 0 and len(utms) != 0)

            # NOTE load query data: using all of them
            qData = {'keys': keys, 'utms': utms}
            keysQ = qData['keys']
            utmQ = qData['utms']
            # NOTE load database data: using all of them
            dbData = {'keys': keys, 'utms': utms}
            keysDb = dbData['keys']
            utmDb = dbData['utms']

            self.qImages.extend(qData['keys'])
            self.dbImages.extend(dbData['keys'])

            self.qEndPosList.append(len(qData['keys']))
            self.dbEndPosList.append(len(dbData['keys']))

            # find positive images for training
            neigh = NearestNeighbors(algorithm='kd_tree')
            print(f'Construct neighbor searches: {neigh.algorithm}')
            neigh.fit(utmDb)
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

                    self.pIdx.append(p_idxs + _lenDb)
                    self.qIdx.append(q_idx + _lenQ)

                    # in training we have two thresholds, one for finding positives and one for finding images
                    # that we are certain are negatives.
                    if self.mode == 'train':

                        n_idxs = nnI[q_idx]
                        self.nonNegIdx.append(n_idxs + _lenDb)

        # when GPS / UTM / pano info is not available
        elif self.mode in ['test']:

            # load images for subtask
            test_keys, test_utms = self.read_info(data_info_path, 'test')
            assert(len(test_keys) != 0 and len(test_utms) != 0)

            # load query data: using all of them
            qData = {'keys': test_keys, 'utms': test_utms}
            keysQ = qData['keys']
            utmQ = qData['utms']
            # load database data: using all of them
            dbData = {'keys': test_keys, 'utms': test_utms}
            keysDb = dbData['keys']
            utmDb = dbData['utms']

            self.qImages.extend(keysQ)
            self.dbImages.extend(keysDb)

            # add query index
            self.qIdx.extend(list(range(_lenQ, len(keysQ) + _lenQ)))

        if len(self.qImages) == 0 or len(self.dbImages) == 0:
            print("Exiting...")
            print("No query/database images.")
            sys.exit()

        # cast to np.arrays for indexing during training
        self.qIdx = np.asarray(self.qIdx)
        self.qImages = np.asarray(self.qImages)
        self.pIdx = np.asarray(self.pIdx, dtype=object)
        self.nonNegIdx = np.asarray(self.nonNegIdx, dtype=object)
        self.dbImages = np.asarray(self.dbImages)

        # decide device type ( important for triplet mining )
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.threads = threads
        self.bs = bs

        if mode == 'train':

            # for now always 1-1 lookup.
            self.negCache = np.asarray([np.empty((0,), dtype=int)] * len(self.qIdx))

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

    def new_epoch(self):
        """Divide the dataset as subsets 
        
        Note: expected to be launched at the beginnng of each epoch
        """

        # find how many subset we need to do 1 epoch
        self.nCacheSubset = math.ceil(len(self.qIdx) / self.cached_queries)

        # get all indices
        arr = np.arange(len(self.qIdx))

        # apply positive sampling of indices
        arr = random.choices(arr, self.weights, k=len(arr))

        # calculate the subcache indices
        self.subcache_indices = np.array_split(arr, self.nCacheSubset)

        # reset subset counter
        self.current_subset = 0

    def update_subcache_vanilla(self):
        qidxs = np.random.choice(len(self.qIdx), self.cached_queries, replace=False)

        for q in qidxs:

            # get query idx
            qidx = self.qIdx[q]

            # get negatives
            while True:
                nidxs = np.random.choice(len(self.dbImages), size=self.nNeg)

                # ensure that non of the choice negative images are within the negative range (default 25 m)
                if sum(np.in1d(nidxs, self.nonNegIdx[q])) == 0:
                    break

            # package the triplet and target
            triplet = [qidx, *nidxs]
            target = [-1] + [0] * len(nidxs)

            self.triplets.append((triplet, target))

        # increment subset counter
        self.current_subset += 1

        return

    def update_subcache(self, net=None, outputdim=None):
        """Prepare current subset's triplets (subcache)

        If `net` is None, randomly choose the subset and compose its triplets.
        The query's positive and negatives are chosen randomly within its
        candidates
        If `net` is not None, take the corresponding indices of the current
        subset id, calculate hard triplets. The query's positive is the closest 
        one, and negative is the hardest nNegs (default: 5) for the input model
        """

        # reset triplets
        self.triplets = []

        # if there is no network associate to the cache, then we don't do any hard negative mining.
        # Instead we just create some naive triplets based on distance.
        # NOTE: in this case, the prepared subcache_indices is not utilized
        if net is None:
            self.update_subcache_vanilla()
            return

        # take n query images
        if self.current_subset >= len(self.subcache_indices):
            tqdm.write('Reset epoch - FIX THIS LATER!')
            self.current_subset = 0
        qidxs = np.asarray(self.subcache_indices[self.current_subset])

        # take m = 5*cached_queries is number of negative images
        nidxs = np.random.choice(len(self.dbImages), self.cached_negatives, replace=False)

        # and make sure that there is no positives among them
        nidxs = nidxs[np.in1d(nidxs, np.unique([i for idx in self.nonNegIdx[qidxs] for i in idx]), invert=True)]

        # make dataloaders for query, positive and negative images
        opt = {'batch_size': self.bs, 'shuffle': False, 'num_workers': self.threads, 'pin_memory': True, 'collate_fn': ImagePairsFromList.collate_fn}
        qloader = torch.utils.data.DataLoader(ImagePairsFromList(self.root_dir, self.qImages[qidxs], transform=self.transform), **opt)
        nloader = torch.utils.data.DataLoader(ImagePairsFromList(self.root_dir, self.dbImages[nidxs], transform=self.transform), **opt)

        # calculate their descriptors
        net.eval()
        with torch.no_grad():

            # initialize descriptors
            qvecs_gr = torch.zeros(len(qidxs), outputdim).to(self.device)
            qvecs_sa = torch.zeros(len(qidxs), outputdim).to(self.device)

            nvecs_gr = torch.zeros(len(nidxs), outputdim).to(self.device)
            nvecs_sa = torch.zeros(len(nidxs), outputdim).to(self.device)

            bs = opt['batch_size']

            # compute descriptors
            for i, batch in tqdm(enumerate(qloader), desc='compute query descriptors', total=len(qidxs) // bs,
                                 position=2, leave=False):
                if batch is None:
                    # NOTE: if the batch is unavailable (e.g., a sample in it meets FileNotFoundError),
                    # simply leave the query and positive desc values as zeros. If unluckily they are
                    # chosen, we will handle the FileNotFoundError again in the main loop
                    continue
                X, y = batch
                encoding = net(*[x.to(self.device) for x in X])
                # NOTE: the net is expected to return a tuple (desc ground, desc satellite)
                qvecs_gr[i * bs:(i + 1) * bs, :] = encoding[0]
                qvecs_sa[i * bs:(i + 1) * bs, :] = encoding[1]
            for i, batch in tqdm(enumerate(nloader), desc='compute negative descriptors', total=len(nidxs) // bs,
                                 position=2, leave=False):
                if batch is None:
                    continue
                X, y = batch
                encoding = net(*[x.to(self.device) for x in X])
                nvecs_gr[i * bs:(i + 1) * bs, :] = encoding[0]
                nvecs_sa[i * bs:(i + 1) * bs, :] = encoding[1]

        tqdm.write('>> Searching for hard negatives...')
        # compute dot product scores and ranks on GPU
        # NOTE: current strategy to calculate score between samples (each sample is a pair)

        # calculate distance between query and negatives
        nScores = torch.mm(qvecs_gr, nvecs_sa.t()) + torch.mm(qvecs_sa, nvecs_gr.t()) / 2
        nScores, nRanks = torch.sort(nScores, dim=1, descending=True)

        # convert to cpu and numpy
        nScores, nRanks = nScores.cpu().numpy(), nRanks.cpu().numpy()

        # selection of hard triplets
        for q in range(len(qidxs)):

            qidx = qidxs[q]

            # take the score with itself 
            dPos = np.sum((qvecs_gr[q] * qvecs_sa[q]).cpu().numpy())
            
            # get distances to all negatives
            dNeg = nScores[q, :]

            # how much are they violating # NOTE: violating means there're space for them to improve. We want them now
            loss = dPos - dNeg + self.margin ** 0.5
            violatingNeg = 0 < loss

            # if less than nNeg are violating then skip this query
            if np.sum(violatingNeg) <= self.nNeg:
                continue

            # select hardest negatives
            hardest_negIdx = np.argsort(loss)[:self.nNeg]

            # select the hardest negatives
            cached_hardestNeg = nRanks[q, hardest_negIdx]

            # transform back to original index (back to original idx domain)
            qidx = self.qIdx[qidx]
            hardestNeg = nidxs[cached_hardestNeg]

            # package the triplet and target
            triplet = [qidx, *hardestNeg]
            target = [-1] + [0] * len(hardestNeg)

            self.triplets.append((triplet, target))

        # increment subset counter
        self.current_subset += 1

    def __getitem__(self, idx):
        # get triplet
        triplet, target = self.triplets[idx]

        # get query, positive and negative idx
        qidx = triplet[0]
        nidx = triplet[1:]

        keys = {}
        keys['query'] = self.qImages[qidx]
        keys['negatives'] = [self.dbImages[idx] for idx in nidx]

        # load images into triplet list
        try:
            query_gr = self.transform(Image.open(join(self.gr_path, keys['query']['gr_img'])))
            query_sa = self.transform(Image.open(join(self.sa_path, keys['query']['sa_img'])))
            query = query_gr, query_sa

            negatives_gr = [self.transform(Image.open(join(self.gr_path, k['gr_img']))) 
                            for k in keys['negatives']]
            negatives_gr = torch.stack(negatives_gr, 0)
            negatives_sa = [self.transform(Image.open(join(self.sa_path, k['sa_img']))) 
                            for k in keys['negatives']]
            negatives_sa = torch.stack(negatives_sa, 0)
            negatives = negatives_gr, negatives_sa
        except:
            # NOTE: errors met till now: 
            # FileNotFoundError, OSError: image file is truncated
            # https://stackoverflow.com/a/23575424 could solve it but we choose not to use it
            return None

        return query, negatives, [qidx] + nidx, keys

    def __len__(self):
        return len(self.triplets)

    @staticmethod
    def collate_fn(batch):
        """Creates mini-batch tensors from the list of tuples (query, positive, negatives).

        Args:
            batch: list of tuple (query, positive, negatives).
                - query: torch tensor of shape (3, h, w).
                - positive: torch tensor of shape (3, h, w).
                - negative: torch tensor of shape (n, 3, h, w).
        Returns:
            query: torch tensor of shape (batch_size, 3, h, w).
            positive: torch tensor of shape (batch_size, 3, h, w).
            negatives: torch tensor of shape (batch_size, n, 3, h, w).
        """
        if None in batch:
            return None

        query, negatives, indices, keys = zip(*batch)

        query_gr = data.dataloader.default_collate([q[0] for q in query])
        query_sa = data.dataloader.default_collate([q[1] for q in query])
        query = query_gr, query_sa

        negCounts = data.dataloader.default_collate([x[0].shape[0] for x in negatives])

        negatives_gr = torch.cat([n[0] for n in negatives], 0)
        negatives_sa = torch.cat([n[1] for n in negatives], 0)
        negatives = negatives_gr, negatives_sa

        indices = list(itertools.chain(*indices))
        keys = list(keys)

        meta = {
            'negCounts': negCounts,
            'indices': indices,
            'keys': keys
        }

        return query, negatives, meta