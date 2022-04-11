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
from orissl_cvm.utils import input_transform
from orissl_cvm.tools.visualize import visualize_desc


def val(eval_set, model, desc_dim, device, config, writer, epoch_num=0, write_tboard=False, pbar_position=0):
    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False
    eval_set_queries = ImagePairsFromList(eval_set.root_dir, eval_set.qImages, transform=input_transform())
    opt = {
        'batch_size': config.train.batch_size, 
        'shuffle': False, 
        'num_workers': int(config.train.threads), 
        'pin_memory': cuda, 
        'collate_fn': ImagePairsFromList.collate_fn
    }
    test_data_loader_queries = DataLoader(dataset=eval_set_queries, **opt)

    model.eval()
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        qFeat_gr = np.empty((len(eval_set_queries), desc_dim), dtype=np.float32)
        qFeat_sa = np.empty((len(eval_set_queries), desc_dim), dtype=np.float32)

        for iteration, batch in enumerate(tqdm(test_data_loader_queries, 
                position=pbar_position, leave=False, desc='Test Iter'.rjust(15)), 1):
            if batch is None: 
                tqdm.write('====> Batch data iteration is None. Probably caused by corrupted file')
                continue
            img_gr, img_sa, indices = batch
            img_gr, img_sa = img_gr.to(device), img_sa.to(device)
            encoding = model(img_gr, img_sa)
            qFeat_gr[indices.detach().numpy(), :] = encoding[0].detach().cpu().numpy()
            qFeat_sa[indices.detach().numpy(), :] = encoding[1].detach().cpu().numpy()

            del img_gr, img_sa, encoding

    del test_data_loader_queries

    # NOTE visualize 8 samples in val set for debug
    # visualize_desc(torch.from_numpy(qFeat_gr)[:8], torch.from_numpy(qFeat_sa)[:8])

    tqdm.write('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20, 50, 100]
    # NOTE for debug on a single 8-sample batch
    # n_values = [1, 2, 3, 4, 5, 6, 7, 8]

    # for each query get those within threshold distance
    gt = eval_set.all_pos_indices

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
    recall_at_n = correct_at_n / len(eval_set.qIdx)

    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        tqdm.write("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard:
            writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch_num)

    return all_recalls
