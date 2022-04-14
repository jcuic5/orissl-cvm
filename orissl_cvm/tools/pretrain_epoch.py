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
from orissl_cvm.utils import soft_triplet_loss
from orissl_cvm.datasets.cvact_dataset import CVACTDataset
from orissl_cvm.tools.visualize import visualize_scores, visualize_plain_batch_pretrain


def pretrain_epoch(train_dataset, training_data_loader, model, 
                optimizer, criterion, encoder_dim, device, 
                epoch_num, config, writer):
        
    epoch_loss = 0
    n_batches = (len(train_dataset.qIdx) + config.train.batch_size - 1) // config.train.batch_size

    model.train()
    for iteration, batch in enumerate(tqdm(training_data_loader, 
                                            position=1, 
                                            leave=False, 
                                            desc='Train Iter'.rjust(15))):
        # in case we get an empty batch
        if batch is None:
            tqdm.write('====> Batch data iteration is None. Probably caused by corrupted file')
            continue

        # unwrap the batch information
        query_gr, query_sa, label, meta = batch
        # NOTE replace the satellite by another one, for debug
        # query_sa[...] = 0

        indices, keys = meta['indices'], meta['keys']
        B = query_sa.shape[0]
        query_gr, query_sa, label = query_gr.to(device), query_sa.to(device), label.to(device)

        # forward
        output = model(query_gr, query_sa)
        _, predicted = torch.max(output, dim=1)

        # NOTE visualize the descriptor for debug
        # if i % 50 == 0:
            # visualize_desc(descQ_gr, descQ_sa)

        # calculate loss, back propagate, update weights
        optimizer.zero_grad()
        loss = criterion(output, label)
        loss /= B
        loss.backward()
        optimizer.step()

        # NOTE visualize batch and score for debug
        # visualize_plain_batch_pretrain(batch)
        # visualize_scores(output, label)

        del query_gr, query_sa

        batch_loss = loss.item()
        epoch_loss += batch_loss

        if iteration % 25 == 0 or n_batches <= 10:
            tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}, Accuracy: {:.4f}".format(
                epoch_num, iteration, n_batches, batch_loss, (predicted == label).sum().item() / B))
            writer.add_scalar('Train/Loss', batch_loss,
                                ((epoch_num - 1) * n_batches) + iteration)

    avg_loss = epoch_loss / n_batches
    tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, avg_loss))
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)

    tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
    tqdm.write('Cached:    ' + humanbytes(torch.cuda.memory_reserved()))