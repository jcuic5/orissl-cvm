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


import torch
from tqdm.auto import trange, tqdm
from torch.utils.data import DataLoader
from orissl_cvm.tools import humanbytes
from orissl_cvm.datasets.cvact_dataset import CVACTDataset
from orissl_cvm.tools.visualize import visualize_desc


def train_epoch(train_dataset, training_data_loader, model, 
                optimizer, scheduler, criterion, device, 
                epoch_num, cfg, writer):
        
    epoch_loss = 0
    n_batches = (len(train_dataset.qIdx) + cfg.train.batch_size - 1) // cfg.train.batch_size

    model.train()
    local_progress = tqdm(training_data_loader, position=1, leave=False, desc='Train Iter'.rjust(15))
    for iteration, batch in enumerate(local_progress):
        # prepare data
        if batch is None:
            tqdm.write('==> Current batch is None')
            continue
        input_data, meta = batch
        query_gr, query_sa = input_data
        indices, keys_gr, keys_sa = meta['indices'], meta['keys_gr'], meta['keys_sa']
        qpn_mat = torch.from_numpy(train_dataset.qpn_matrix[indices, :][:, indices]).to(device)
        qn_triplets = torch.nonzero(qpn_mat == 0, as_tuple=False)
        n_triplets = qn_triplets.shape[0]
        if n_triplets == 0:
            del query_gr, query_sa
            continue
        
        # forward
        query_gr, query_sa = query_gr.to(device), query_sa.to(device)
        descQ_gr, descQ_sa = model(query_gr, query_sa)
        # NOTE visualize the descriptor for debug
        # if i % 50 == 0:
            # visualize_desc(descQ_gr, descQ_sa)

        # calculate loss, back propagate, update weights
        optimizer.zero_grad()
        loss = 0
        for i in range(n_triplets):
            qidx, nidx = qn_triplets[i][0].item(), qn_triplets[i][1].item()
            loss += criterion(descQ_gr[qidx: qidx+1], descQ_sa[qidx: qidx+1], descQ_sa[nidx: nidx+1])
            loss += criterion(descQ_sa[qidx: qidx+1], descQ_gr[qidx: qidx+1], descQ_gr[nidx: nidx+1])
        loss /= n_triplets # normalise by actual number of negatives
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        del query_gr, query_sa, descQ_gr, descQ_sa

        batch_loss = loss.item()
        epoch_loss += batch_loss
        if iteration % (n_batches // 5) == 0 or n_batches <= 10:
            tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch_num, iteration,
                                                                    n_batches, batch_loss))
            writer.add_scalar('Train/Loss', batch_loss,
                                ((epoch_num - 1) * n_batches) + iteration)

    avg_loss = epoch_loss / n_batches
    tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, avg_loss))
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)

    # tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
    # tqdm.write('Cached:    ' + humanbytes(torch.cuda.memory_reserved()))