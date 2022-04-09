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
'''


import torch
from tqdm.auto import trange, tqdm
from torch.utils.data import DataLoader
from orissl_cvm.utils.tools import humanbytes, soft_triplet_loss
from orissl_cvm.datasets.cvact_dataset import CVACTDataset
from orissl_cvm.utils.visualize import visualize_desc


def train_epoch(train_dataset, training_data_loader, model, 
                optimizer, criterion, encoder_dim, device, 
                epoch_num, opt, config, writer):
        
    epoch_loss = 0
    nBatches = (len(train_dataset.qIdx) + int(config['train']['batchsize']) - 1) // \
                                                    int(config['train']['batchsize'])

    model.train()
    # Iterate
    for iteration, batch in enumerate(tqdm(training_data_loader, 
                                            position=1, 
                                            leave=False, 
                                            desc='Train Iter'.rjust(15))):
        for i in range(100):
            # in case we get an empty batch
            if batch is None:
                continue

            # unwrap the batch information
            query, meta = batch
            indices, keys, qpn_mat = meta['indices'], meta['keys'], meta['qpn_mat']
            B = query[0].shape[0]
            data_input = [x.to(device) for x in list(query)]
            qpn_mat = torch.from_numpy(qpn_mat).to(device)

            # forward
            descQ_gr, descQ_sa = model(*data_input)

            # if i % 50 == 0:
                # visualize_desc(descQ_gr, descQ_sa)

            # calculate loss, back propagate, update weights
            optimizer.zero_grad()
            loss = 0
            qn_triplets = torch.nonzero(qpn_mat == 0, as_tuple=False)
            nTriplets = qn_triplets.shape[0]
            for i in range(nTriplets):
                qidx, nidx = qn_triplets[i][0].item(), qn_triplets[i][1].item()
                loss += criterion(descQ_gr[qidx: qidx+1], descQ_sa[qidx: qidx+1], descQ_sa[nidx: nidx+1])
                loss += criterion(descQ_sa[qidx: qidx+1], descQ_gr[qidx: qidx+1], descQ_gr[nidx: nidx+1])
                # loss += soft_triplet_loss(descQ_gr[qidx: qidx+1], descQ_sa[qidx: qidx+1], descQ_sa[nidx: nidx+1])
                # loss += soft_triplet_loss(descQ_sa[qidx: qidx+1], descQ_gr[qidx: qidx+1], descQ_gr[nidx: nidx+1])
            loss /= nTriplets # normalise by actual number of negatives
            loss.backward()
            optimizer.step()

            del data_input, descQ_gr, descQ_sa,
            del query

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 25 == 0 or nBatches <= 10:
                tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch_num, iteration,
                                                                        nBatches, batch_loss))
                writer.add_scalar('Train/Loss', batch_loss,
                                    ((epoch_num - 1) * nBatches) + iteration)

    avg_loss = epoch_loss / nBatches
    tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, avg_loss))
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)

    tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
    tqdm.write('Cached:    ' + humanbytes(torch.cuda.memory_reserved()))