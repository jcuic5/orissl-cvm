import imp
from os.path import join
from PIL import Image
import torch.utils.data as data
from orissl_cvm.datasets.cvact_dataset import CVACTDataset
import random
import torch


def random_slide_pano(img):
    H, W = img.shape[-2], img.shape[-1]
    label = random.randint(0, 35)
    slide_w = int(float(label / 36) * W)
    img = torch.cat([img[..., slide_w:], img[..., :slide_w]], dim=-1)
    label = torch.zeros(36, dtype=torch.float).scatter_(dim=0, index=torch.tensor(label), value=1)
    return img, label


class CVACTDatasetPretrain(CVACTDataset):
    def __init__(self, root_dir, mode='train', transform=None, posDistThr=5, negDistThr=100, positive_sampling=False, mini_scale=0.25):
        super().__init__(root_dir, mode, transform, posDistThr, negDistThr, positive_sampling, mini_scale)
    
    def __getitem__(self, qidx):
        key = self.qImages[qidx]
        # load images
        query_gr, label = random_slide_pano(self.transform(Image.open(join(self.gr_path, key['gr_img']))))
        query_sa = self.transform(Image.open(join(self.sa_path, key['sa_img'])))
        return query_gr, query_sa, label, key, qidx

    @staticmethod
    def collate_fn(batch):
        if None in batch:
            return None
        query_gr, query_sa, label, key, qidx = zip(*batch)
        query_gr = data.dataloader.default_collate(query_gr)
        query_sa = data.dataloader.default_collate(query_sa)
        label = data.dataloader.default_collate(label)
        qidx, key = list(qidx), list(key)
        meta = {
            'indices': qidx,
            'keys': key
        }
        return query_gr, query_sa, label, meta