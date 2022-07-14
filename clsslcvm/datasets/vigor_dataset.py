import imp
import os
from matplotlib.pyplot import axis
import numpy as np
import random
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import BatchSampler


class VIGORDataset(Dataset):
    def __init__(self, root_dir, mode, transform, logger, version='', dim=4096, same_area=False, continuous=False):
        assert mode in ['train', 'val']
        self.root_dir = root_dir
        self.mode = mode
        self.logger = logger
        self.dim = dim
        self.same_area = same_area
        self.continuous = continuous
        self.label_root = 'splits' if version == '' else f'splits_{version}'

        self.sat_size = [224, 224] # [320, 320]
        self.grd_size = [224, 448] # [320, 640]
        self.tf_sat = transform(self.sat_size)
        self.tf_grd = transform(self.grd_size)

        # full set
        if version == 'newfull':
            if self.same_area:
                self.city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
            else:
                if self.mode == 'train':
                    # self.city_list = ['NewYork', 'Seattle']
                    self.city_list = ['SanFrancisco', 'Chicago'] #! swapped split
                else:
                    # self.city_list = ['SanFrancisco', 'Chicago']
                    self.city_list = ['NewYork', 'Seattle'] #! swapped split
        # partial set
        else:
            if self.same_area:
                self.city_list = ['Chicago']
            else:
                if self.mode == 'train':
                    self.city_list = ['Chicago']
                else:
                    self.city_list = ['Chicago']

        info_dict = self.read_info()
        self.sat_list = info_dict['sat_list']
        self.sat_index_dict = info_dict['sat_index_dict']
        self.sat_list_size = info_dict['sat_list_size']
        self.grd_list = info_dict['grd_list']
        self.grd_sat_label = info_dict['grd_sat_label']
        self.sat_cover_dict = info_dict['sat_cover_dict']
        self.grd_sat_delta = info_dict['grd_sat_delta']
        self.grd_list_size = info_dict['grd_list_size']
        self.sat_cover_list = info_dict['sat_cover_list']
        
        # only for analysis
        self.mean_product = 0.
        self.mean_positive_product = 0.7
        self.mean_hit = 0.5


    def read_info(self):
        sat_list = []
        sat_index_dict = {}
        idx = 0
        # load sat list
        for city in self.city_list:
            sat_list_fname = os.path.join(self.root_dir, self.label_root, city, 'satellite_list.txt')
            with open(sat_list_fname, 'r') as file:
                for line in file.readlines():
                    sat_list.append(os.path.join(self.root_dir, city, 'satellite', line.replace('\n', '')))
                    sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            self.logger.info(f'VIGOR dataset: load {sat_list_fname}: {idx}')
        sat_list_size = len(sat_list)
        sat_list = np.array(sat_list)
        self.logger.info('{} sat loaded, data size:{}'.format(self.mode, sat_list_size))

        grd_list = []
        grd_sat_label = []
        sat_cover_dict = {}
        grd_sat_delta = []
        idx = 0
        for city in self.city_list:
            # load train panorama list
            label_fname = os.path.join(self.root_dir, self.label_root, city, 'same_area_balanced_train.txt'
                                       if self.same_area else 'pano_label_balanced.txt')
            with open(label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(sat_index_dict[data[i]])
                    label = np.array(label).astype(np.int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    grd_list.append(os.path.join(self.root_dir, city, 'panorama', data[0]))
                    grd_sat_label.append(label)
                    grd_sat_delta.append(delta)
                    if not label[0] in sat_cover_dict:
                        sat_cover_dict[label[0]] = [idx]
                    else:
                        sat_cover_dict[label[0]].append(idx)
                    idx += 1
            self.logger.info(f'VIGOR dataset: load {label_fname}: {idx}')
        grd_list_size = len(grd_list)
        grd_sat_label = np.array(grd_sat_label)
        grd_sat_delta = np.array(grd_sat_delta)
        self.logger.info('{} grd loaded, data_size: {}'.format(self.mode, grd_list_size))

        sat_cover_list = list(sat_cover_dict.keys())

        info_dict = {
            'sat_list': sat_list,
            'sat_index_dict': sat_index_dict,
            'sat_list_size': sat_list_size,
            'grd_list': grd_list,
            'grd_sat_label': grd_sat_label,
            'sat_cover_dict': sat_cover_dict,
            'grd_sat_delta': grd_sat_delta,
            'grd_list_size': grd_list_size,
            'sat_cover_list': sat_cover_list
        }
        return info_dict


    def __getitem__(self, index):
        try:
            image_sat = self.read_image(self.sat_list[self.grd_sat_label[index][0]])
            image_grd = self.read_image(self.grd_list[index])
            if self.continuous:
                randx = random.randrange(1, 4)
                image_sat_semi = self.read_image(self.sat_list[self.grd_sat_label[index][randx]])
        except (FileNotFoundError, OSError):
            image_sat = Image.new('RGB', (self.sat_size[0], self.sat_size[1]))
            image_grd = Image.new('RGB', (self.grd_size[0], self.grd_size[1]))
            if self.continuous:
                image_sat_semi = Image.new('RGB', (self.sat_size[0], self.sat_size[1]))
        image_sat = self.tf_sat(image_sat)
        image_grd = self.tf_grd(image_grd)
        if self.continuous:
            image_sat_semi = self.tf_sat(image_sat_semi)
            return image_sat, image_sat_semi, image_grd, index, self.grd_sat_delta[index, 0], self.grd_sat_delta[index, randx]
        return image_sat, image_grd, index, self.grd_sat_delta[index, 0]


    def __len__(self):
        return self.grd_list_size


    def read_image(self, path):
        # img = cv2.imread(path).astype(np.float32)
        # img[:, :, 0] -= 103.939  # Blue
        # img[:, :, 1] -= 116.779  # Green
        # img[:, :, 2] -= 123.6  # Red
        # img = Image.fromarray(img.astype(np.uint8))
        img = Image.open(path).convert("RGB")
        return img


class NonOverlapBatchSampler(BatchSampler):
    def __init__(self, sampler, label, batch_size, drop_last):
        super().__init__(sampler, batch_size, drop_last)
        self.label = label
        self.cache = []

    def __iter__(self):
        batch = []
        for idx in self.sampler:
            if self.check_non_overlap(batch, idx):
                batch.append(idx)
            else:
                for i in self.cache:
                    if self.check_non_overlap(batch, i):
                        batch.append(i)
                        self.cache.remove(i)
                        break
                self.cache.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        for idx in self.cache.copy():
            if self.check_non_overlap(batch, idx):
                batch.append(idx)
                self.cache.remove(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch) > 1 and not self.drop_last: # > 1 to ensure there's at least one triplet
            yield batch

    def check_non_overlap(self, id_list, idx):
        output = True
        sat_idx = self.label[idx]
        for id in id_list:
            sat_id = self.label[id]
            for i in sat_id:
                if i in sat_idx:
                    output = False
                    return output
        return output