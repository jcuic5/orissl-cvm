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

Significant parts of our code are based on [Nanne's pytorch-netvlad repository]
(https://github.com/Nanne/pytorch-NetVlad/), as well as some parts from the [Mapillary SLS repository]
(https://github.com/mapillary/mapillary_sls)

Modified by Jianfeng Cui
'''


import numpy as np
import torch
import faiss
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from orissl_cvm.datasets.cvact_dataset import ImagePairsFromList
from orissl_cvm.augmentations import input_transform
from orissl_cvm.tools.visualize import visualize_desc

def val(val_dataset, val_dataset_queries, val_dataloader_queries, model, device, writer, epoch_num=0, write_tboard=False, pbar_position=0):
    model.eval()
    # dynamically determine descriptor's dim
    it = iter(val_dataloader_queries)
    with torch.no_grad():
        img_gr, img_sa, indices = next(it)
        img_gr, img_sa = img_gr.to(device), img_sa.to(device)
        desc_dim = model(img_gr, img_sa)[0].shape[-1]
    del img_gr, img_sa, indices, it

    # start extracting features
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        qFeat_gr = np.empty((len(val_dataset_queries), desc_dim), dtype=np.float32)
        qFeat_sa = np.empty((len(val_dataset_queries), desc_dim), dtype=np.float32)

        local_progress = tqdm(val_dataloader_queries, position=pbar_position, leave=False, desc='Test Iter'.rjust(15))
        for iteration, batch in enumerate(local_progress, 1):
            if batch is None: 
                tqdm.write('====> Current batch is None')
                continue
            img_gr, img_sa, indices = batch
            img_gr, img_sa = img_gr.to(device), img_sa.to(device)
            descQ_gr, descQ_sa = model(img_gr, img_sa)
            qFeat_gr[indices.detach().numpy(), :] = descQ_gr.detach().cpu().numpy()
            qFeat_sa[indices.detach().numpy(), :] = descQ_sa.detach().cpu().numpy()

            del img_gr, img_sa, descQ_gr, descQ_sa

    # NOTE visualize 8 samples in val set for debug
    # visualize_desc(torch.from_numpy(qFeat_gr)[:8], torch.from_numpy(qFeat_sa)[:8])

    # start evaluation
    tqdm.write('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20, 50, 100]
    # NOTE for debug on a single 8-sample batch
    # n_values = [1, 2, 3, 4, 5, 6, 7, 8]

    # for each query get those within threshold distance
    gt = val_dataset.all_pos_indices

    # any combination of mapillary cities will work as a val set
    faiss_index = faiss.IndexFlatL2(desc_dim)
    faiss_index.add(qFeat_sa)
    _, predictions = faiss_index.search(qFeat_gr, max(n_values))

    correct_at_n = np.zeros(len(n_values))
    # TODO can we do this on the matrix in one go?
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    # 简单来说，就是在以这个严格标准下评判（只要pred中的前这些个里面有一个真的是gt里的
    # 一个就可以）总共有多少比例的qidx
    recall_at_n = correct_at_n / len(val_dataset.qIdx)

    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        tqdm.write("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard:
            writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch_num)

    return all_recalls
