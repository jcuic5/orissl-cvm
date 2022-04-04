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
from orissl_cvm.utils.tools import humanbytes
from orissl_cvm.datasets.cvact_dataset import CVACTDataset


def train_epoch(train_dataset, model, optimizer, criterion, 
                encoder_dim, device, epoch_num, 
                opt, config, writer):

    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False

    # Divide the dataset as subsets (by indices)
    train_dataset.new_epoch()
    print(f'Full num of queries: {len(train_dataset.qIdx)}')
    print(f'Size of subsets: {train_dataset.cached_queries}')
    print(f'Num of subsets: {len(train_dataset.subcache_indices)}')

    epoch_loss = 0
    startIter = 1  # keep track of batch iter across subsets for logging

    nBatches = (len(train_dataset.qIdx) + int(config['train']['batchsize']) - 1) // int(config['train']['batchsize'])

    # Iterate over subsets
    for subIter in trange(train_dataset.nCacheSubset, desc='Cache refresh'.rjust(15), position=1):
        pool_size = encoder_dim

        tqdm.write('====> Building Cache')
        # Prepare current triplets (subset, subcache) in the dataset 
        train_dataset.update_subcache(model, pool_size)

        # Dataloader
        training_data_loader = DataLoader(dataset=train_dataset, num_workers=opt.threads,
                                          batch_size=int(config['train']['batchsize']), shuffle=True,
                                          collate_fn=CVACTDataset.collate_fn, pin_memory=cuda)

        tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
        tqdm.write('Cached:    ' + humanbytes(torch.cuda.memory_reserved()))

        model.train()
        
        # Iterate
        for iteration, batch in enumerate(tqdm(training_data_loader, 
                                               position=2, 
                                               leave=False, 
                                               desc='Train Iter'.rjust(15)), 
                                               startIter):
            # in case we get an empty batch
            if batch is None:
                continue

            # Unwrap the batch information
            query, positive, negatives, meta = batch
            negCounts, indices, keys = meta['negCounts'], meta['indices'], meta['keys']

            # Prepare batch for input
            # some reshaping to put query, pos, negs in a single (N, 3, H, W) tensor
            # where N = batchSize * (nQuery + nPos + nNeg)
            B = query[0].shape[0]
            nNeg = torch.sum(negCounts)
            triplets_gr = torch.cat([query[0], positive[0], negatives[0]])
            triplets_sa = torch.cat([query[1], positive[1], negatives[1]])
            data_input = [triplets_gr, triplets_sa]

            data_input = [x.to(device) for x in data_input]

            # Forward
            encoding = model(*data_input)

            descQ_gr, descP_gr, descN_gr = torch.split(encoding[0], [B, B, nNeg])
            descQ_sa, descP_sa, descN_sa = torch.split(encoding[1], [B, B, nNeg])
            optimizer.zero_grad()

            # calculate loss for each Query, Positive, Negative triplet
            # due to potential difference in number of negatives have to
            # do it per query, per negative
            loss = 0
            for i, negCount in enumerate(negCounts):
                for n in range(negCount):
                    negIx = (torch.sum(negCounts[:i]) + n).item()
                    loss += criterion(descQ_gr[i: i + 1], descP_sa[i: i + 1], descN_sa[negIx:negIx + 1])
                    loss += criterion(descQ_sa[i: i + 1], descP_gr[i: i + 1], descN_gr[negIx:negIx + 1])

            loss /= nNeg.float().to(device)  # normalise by actual number of negatives
            loss.backward()
            optimizer.step()
            del data_input, encoding, descQ_gr, descP_gr, descN_gr, descQ_sa, descP_sa, descN_sa
            del query, positive, negatives

            batch_loss = loss.item()
            epoch_loss += batch_loss

            if iteration % 25 == 0 or nBatches <= 10:
                tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch_num, iteration,
                                                                       nBatches, batch_loss))
                writer.add_scalar('Train/Loss', batch_loss,
                                  ((epoch_num - 1) * nBatches) + iteration)
                writer.add_scalar('Train/nNeg', nNeg,
                                  ((epoch_num - 1) * nBatches) + iteration)
                tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
                tqdm.write('Cached:    ' + humanbytes(torch.cuda.memory_reserved()))

        startIter += len(training_data_loader)
        del training_data_loader, loss
        optimizer.zero_grad()
        torch.cuda.empty_cache()

    avg_loss = epoch_loss / nBatches

    tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, avg_loss))
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)
