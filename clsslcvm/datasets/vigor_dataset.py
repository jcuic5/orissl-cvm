import os
from matplotlib.pyplot import axis
import numpy as np
import random
import torch
from PIL import Image

class VIGORDataloader():
    def __init__(self, root_dir, transform, logger, version='', dim=4096, same_area=False, continuous=False, mining=False):
        self.root_dir = root_dir
        self.logger = logger
        self.dim = dim
        self.same_area = same_area
        self.continuous = continuous
        self.mining = mining
        label_root = 'splits' if version == '' else f'splits_{version}'

        self.sat_size = [224, 224] #[320, 320]
        self.grd_size = [224, 448] #[320, 640]
        self.tf_sat = transform(self.sat_size)
        self.tf_grd = transform(self.grd_size)

        if same_area:
            self.train_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
            self.test_city_list = ['NewYork', 'Seattle', 'SanFrancisco', 'Chicago']
        else:
            self.train_city_list = ['NewYork', 'Seattle']
            self.test_city_list = ['SanFrancisco', 'Chicago']

        self.train_sat_list = []
        self.train_sat_index_dict = {}
        self.delta_unit = [0.0003280724526376747, 0.00043301140280175833]
        idx = 0
        # load sat list
        for city in self.train_city_list:
            train_sat_list_fname = os.path.join(self.root_dir, label_root, city, 'satellite_list.txt')
            with open(train_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.train_sat_list.append(os.path.join(self.root_dir, city, 'satellite', line.replace('\n', '')))
                    self.train_sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            self.logger.info(f'VIGORDataloader::__init__: load {train_sat_list_fname}: {idx}')
        self.train_sat_list = np.array(self.train_sat_list)
        self.train_sat_data_size = len(self.train_sat_list)
        self.logger.info('Train sat loaded, data size:{}'.format(self.train_sat_data_size))

        self.test_sat_list = []
        self.test_sat_index_dict = {}
        self.__cur_sat_id = 0  # for test
        idx = 0
        for city in self.test_city_list:
            test_sat_list_fname = os.path.join(self.root_dir, label_root, city, 'satellite_list.txt')
            with open(test_sat_list_fname, 'r') as file:
                for line in file.readlines():
                    self.test_sat_list.append(os.path.join(self.root_dir, city, 'satellite', line.replace('\n', '')))
                    self.test_sat_index_dict[line.replace('\n', '')] = idx
                    idx += 1
            self.logger.info(f'VIGORDataloader::__init__: load {test_sat_list_fname}: {idx}')
        self.test_sat_list = np.array(self.test_sat_list)
        self.test_sat_data_size = len(self.test_sat_list)
        self.logger.info('Test sat loaded, data size:{}'.format(self.test_sat_data_size))

        self.train_list = []
        self.train_label = []
        self.train_sat_cover_dict = {}
        self.train_delta = []
        idx = 0
        for city in self.train_city_list:
            # load train panorama list
            train_label_fname = os.path.join(self.root_dir, label_root, city, 'same_area_balanced_train.txt'
            if self.same_area else 'pano_label_balanced.txt')
            with open(train_label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.train_sat_index_dict[data[i]])
                    label = np.array(label).astype(np.int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.train_list.append(os.path.join(self.root_dir, city, 'panorama', data[0]))
                    self.train_label.append(label)
                    self.train_delta.append(delta)
                    if not label[0] in self.train_sat_cover_dict:
                        self.train_sat_cover_dict[label[0]] = [idx]
                    else:
                        self.train_sat_cover_dict[label[0]].append(idx)
                    idx += 1
            self.logger.info(f'VIGORDataloader::__init__: load {train_label_fname}: {idx}')
        self.train_data_size = len(self.train_list)
        self.train_label = np.array(self.train_label)
        self.train_delta = np.array(self.train_delta)
        self.logger.info('Train grd loaded, data_size: {}'.format(self.train_data_size))

        self.__cur_test_id = 0
        self.test_list = []
        self.test_label = []
        self.test_sat_cover_dict = {}
        self.test_delta = []
        idx = 0
        for city in self.test_city_list:
            # load test panorama list
            test_label_fname = os.path.join(self.root_dir, label_root, city, 'same_area_balanced_test.txt'
            if self.same_area else 'pano_label_balanced.txt')
            with open(test_label_fname, 'r') as file:
                for line in file.readlines():
                    data = np.array(line.split(' '))
                    label = []
                    for i in [1, 4, 7, 10]:
                        label.append(self.test_sat_index_dict[data[i]])
                    label = np.array(label).astype(np.int)
                    delta = np.array([data[2:4], data[5:7], data[8:10], data[11:13]]).astype(float)
                    self.test_list.append(os.path.join(self.root_dir, city, 'panorama', data[0]))
                    self.test_label.append(label)
                    self.test_delta.append(delta)
                    if not label[0] in self.test_sat_cover_dict:
                        self.test_sat_cover_dict[label[0]] = [idx]
                    else:
                        self.test_sat_cover_dict[label[0]].append(idx)
                    idx += 1
            self.logger.info(f'VIGORDataloader::__init__: load {test_label_fname}: {idx}')
        self.test_data_size = len(self.test_list)
        self.test_label = np.array(self.test_label)
        self.test_delta = np.array(self.test_delta)
        self.logger.info('Test grd loaded, data size: {}'.format(self.test_data_size))

        self.train_sat_cover_list = list(self.train_sat_cover_dict.keys())

        # only for analysis
        self.mean_product = 0.
        self.mean_positive_product = 0.7
        self.mean_hit = 0.5

        # for mining pool
        self.mining_pool_size = 40000
        self.mining_save_size = 100
        self.choice_pool = range(self.mining_save_size)
        if self.mining:
            self.sat_global_train = np.zeros([self.train_sat_data_size, dim])
            self.grd_global_train = np.zeros([self.train_data_size, dim])
            self.mining_save = np.zeros([self.train_data_size, self.mining_save_size])
            self.mining_pool_ready = False


    def next_batch_test_sat(self, batch_size):
        if self.__cur_sat_id >= self.test_sat_data_size:
            self.__cur_sat_id = 0
            return None
        elif self.__cur_sat_id + batch_size >= self.test_sat_data_size:
            batch_size = self.test_sat_data_size - self.__cur_sat_id
        batch_sat = torch.zeros([batch_size, 3, self.sat_size[0], self.sat_size[1]])
        for i in range(batch_size):
            img_idx = self.__cur_sat_id + i
            batch_sat[i, :, :, :] = self.tf_sat(self.read_image(self.test_sat_list[img_idx]))
        self.__cur_sat_id += batch_size
        return batch_sat


    def next_batch_test_grd(self, batch_size):
        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None
        elif self.__cur_test_id + batch_size >= self.test_data_size:
            batch_size = self.test_data_size - self.__cur_test_id
        batch_grd = torch.zeros([batch_size, 3, self.grd_size[0], self.grd_size[1]])
        for i in range(batch_size):
            img_idx = self.__cur_test_id + i
            batch_grd[i, :, :, :] = self.tf_grd(self.read_image(self.test_list[img_idx]))
        self.__cur_test_id += batch_size
        return batch_grd


    # load according to retrieval order, for offset prediction after retrieval, the retrieved one may not be positive
    def next_batch_test_with_order(self, batch_size, order_list):
        if self.__cur_test_id >= self.test_data_size:
            self.__cur_test_id = 0
            return None, None, None
        elif self.__cur_test_id + batch_size >= self.test_data_size:
            batch_size = self.test_data_size - self.__cur_test_id

        batch_list = []
        batch_sat = torch.zeros([batch_size, 3, self.sat_size[0], self.sat_size[1]])
        batch_grd = torch.zeros([batch_size, 3, self.grd_size[0], self.grd_size[1]])
        for i in range(batch_size):
            img_idx = self.__cur_test_id + i
            batch_list.append(img_idx)
            batch_sat[i, :, :, :] = self.tf_sat(self.read_image(self.test_sat_list[order_list[img_idx]]))
            batch_grd[i, :, :, :] = self.tf_grd(self.read_image(self.test_list[img_idx]))
        self.__cur_test_id += batch_size
        return batch_sat, batch_grd, np.array(batch_list)


    # avoid sampling overlap images
    def check_non_overlap(self, id_list, idx):
        output = True
        sat_idx = self.train_label[idx]
        for id in id_list:
            sat_id = self.train_label[id]
            for i in sat_id:
                if i in sat_idx:
                    output = False
                    return output
        return output


    def gen_idx(self):
        # random sampling according to sat
        return random.choice(self.train_sat_cover_dict[random.choice(self.train_sat_cover_list)])


    def next_batch_train(self, batch_size):
        if self.mining and self.mining_pool_ready:
            if self.continuous:
                delta_list = np.ones([batch_size * 2, 2])
                batch_sat = torch.zeros([batch_size, 3, self.sat_size[0], self.sat_size[1]])
                batch_grd = torch.zeros([batch_size, 3, self.grd_size[0], self.grd_size[1]])
                batch_list = []
                for batch_idx in range(int(batch_size / 2)):
                    while True:
                        img_idx = self.gen_idx()
                        if self.check_non_overlap(batch_list, img_idx): break
                    image_sat, image_sat_semi, image_grd, delta, delta_semi = self.get_item(img_idx)
                    batch_sat[batch_idx, :, :, :] = image_sat
                    delta_list[batch_idx, :] = delta
                    batch_sat[batch_idx + batch_size, :, :, :] = image_sat_semi
                    delta_list[batch_idx + batch_size, :] = delta_semi
                    batch_grd[batch_idx, :, :, :] = image_grd
                    batch_list.append(img_idx)

                for batch_idx in range(int(batch_size / 2)):
                    choice_count = 0
                    while True:
                        if choice_count <= len(self.choice_pool):
                            sat_id = self.mining_save[batch_list[batch_idx], -1 - random.choice(self.choice_pool)]
                            if sat_id in self.train_sat_cover_dict:
                                img_idx = random.choice(self.train_sat_cover_dict[sat_id])
                            else:
                                choice_count = choice_count + 1
                                continue
                        else:
                            img_idx = self.gen_idx()
                        choice_count = choice_count + 1
                        if self.check_non_overlap(batch_list, img_idx):
                            break
                    image_sat, image_sat_semi, image_grd, delta, delta_semi = self.get_item(img_idx)
                    batch_sat[int(batch_idx + batch_size / 2), :, :, :] = image_sat
                    delta_list[int(batch_idx + batch_size / 2), :] = delta
                    batch_sat[int(batch_idx + batch_size / 2 + batch_size), :, :, :] = image_sat_semi
                    delta_list[int(batch_idx + batch_size / 2 + batch_size), :] = delta_semi
                    batch_grd[int(batch_idx + batch_size / 2), :, :, :] = image_grd
                    batch_list.append(img_idx)
                return batch_sat, batch_grd, np.array(batch_list), delta_list
            else:
                delta_list = np.ones([batch_size, 2])
                batch_sat = torch.zeros([batch_size, 3, self.sat_size[0], self.sat_size[1]])
                batch_grd = torch.zeros([batch_size, 3, self.grd_size[0], self.grd_size[1]])
                batch_list = []
                for batch_idx in range(int(batch_size / 2)):
                    while True:
                        img_idx = self.gen_idx()
                        if self.check_non_overlap(batch_list, img_idx): break
                    image_sat, image_grd, delta = self.get_item(img_idx)
                    batch_sat[batch_idx, :, :, :] = image_sat
                    batch_grd[batch_idx, :, :, :] = image_grd
                    delta_list[batch_idx, :] = delta
                    batch_list.append(img_idx)
                for batch_idx in range(int(batch_size / 2)):
                    choice_count = 0
                    while True:
                        if choice_count <= len(self.choice_pool):
                            sat_id = self.mining_save[batch_list[batch_idx], -1 - random.choice(self.choice_pool)]
                            if sat_id in self.train_sat_cover_dict:
                                img_idx = random.choice(self.train_sat_cover_dict[sat_id])
                            else:
                                choice_count = choice_count + 1
                                continue
                        else:
                            img_idx = self.gen_idx()
                        choice_count = choice_count + 1
                        if self.check_non_overlap(batch_list, img_idx):
                            break
                    image_sat, image_grd, delta = self.get_item(img_idx)
                    batch_sat[int(batch_idx + batch_size / 2), :, :, :] = image_sat
                    batch_grd[int(batch_idx + batch_size / 2), :, :, :] = image_grd
                    delta_list[int(batch_idx + batch_size / 2), :] = delta
                    batch_list.append(img_idx)
                return batch_sat, batch_grd, np.array(batch_list), delta_list

        else:
            if self.continuous:
                delta_list = np.ones([batch_size * 2, 2])
                batch_sat = torch.zeros([batch_size * 2, 3, self.sat_size[0], self.sat_size[1]])
                batch_grd = torch.zeros([batch_size, 3, self.grd_size[0], self.grd_size[1]])
                batch_list = []
                for batch_idx in range(batch_size):
                    while True:
                        img_idx = self.gen_idx()
                        if self.check_non_overlap(batch_list, img_idx): break
                    image_sat, image_sat_semi, image_grd, delta, delta_semi = self.get_item(img_idx)
                    batch_sat[batch_idx, :, :, :] = image_sat
                    delta_list[batch_idx, :] = delta
                    batch_sat[batch_idx + batch_size, :, :, :] = image_sat_semi
                    delta_list[batch_idx + batch_size, :] = delta_semi
                    batch_grd[batch_idx, :, :, :] = image_grd
                    batch_list.append(img_idx)
                return batch_sat, batch_grd, np.array(batch_list), delta_list
            else:
                delta_list = np.ones([batch_size, 2])
                batch_sat = torch.zeros([batch_size, 3, self.sat_size[0], self.sat_size[1]])
                batch_grd = torch.zeros([batch_size, 3, self.grd_size[0], self.grd_size[1]])
                batch_list = []
                for batch_idx in range(batch_size):
                    while True:
                        img_idx = self.gen_idx()
                        if self.check_non_overlap(batch_list, img_idx): break
                    image_sat, image_grd, delta = self.get_item(img_idx)
                    batch_sat[batch_idx, :, :, :] = image_sat
                    batch_grd[batch_idx, :, :, :] = image_grd
                    delta_list[batch_idx, :] = delta
                    batch_list.append(img_idx)
                return batch_sat, batch_grd, np.array(batch_list), delta_list


    def get_item(self, img_idx):
        image_sat = self.tf_sat(self.read_image(self.train_sat_list[self.train_label[img_idx][0]]))
        image_grd = self.tf_grd(self.read_image(self.train_list[img_idx]))

        if self.continuous:
            randx = random.randrange(1, 4)
            image_sat_semi = self.tf_sat(self.read_image(self.train_sat_list[self.train_label[img_idx][randx]]))
            return image_sat, image_sat_semi, image_grd, self.train_delta[img_idx, 0], self.train_delta[img_idx, randx]

        return image_sat, image_grd, self.train_delta[img_idx, 0]


    def cal_ranking_train_limited(self):
        assert self.mining_pool_size < self.train_sat_data_size
        mining_pool = np.array(random.sample(range(self.train_sat_data_size), self.mining_pool_size))
        product_train = np.matmul(self.grd_global_train, np.transpose(self.sat_global_train[mining_pool, :]))
        product_index = np.argsort(product_train, axis=1)

        for i in range(product_train.shape[0]):
            self.mining_save[i, :] = mining_pool[product_index[i, -self.mining_save_size:]]


    def reset_iter(self):
        self.__cur_test_id = 0
        self.__cur_sat_id = 0


    def read_image(self, path):
        # img = cv2.imread(path).astype(np.float32)
        # img[:, :, 0] -= 103.939  # Blue
        # img[:, :, 1] -= 116.779  # Green
        # img[:, :, 2] -= 123.6  # Red
        # img = Image.fromarray(img.astype(np.uint8))
        img = Image.open(path).convert("RGB")
        return img