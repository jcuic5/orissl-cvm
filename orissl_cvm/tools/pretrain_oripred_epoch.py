import imp
import numpy as np
import torch
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from orissl_cvm.tools import humanbytes
from orissl_cvm.datasets.cvact_dataset import CVACTDataset
import matplotlib.pyplot as plt
from orissl_cvm.tools.visualize import denormalize
from orissl_cvm.loss import cycle_mse_loss


def train_epoch(train_dataset, training_data_loader, model, 
                optimizer, scheduler, criterion, device, 
                epoch_num, config, writer):
    epoch_loss = 0
    n_batches = (len(train_dataset.qIdx) + config.train.batch_size - 1) // config.train.batch_size
    model.train()
    local_progress = tqdm(training_data_loader, position=1, leave=False, desc='Train Iter'.rjust(15))
    for iteration, batch in enumerate(local_progress):
        # prepare data
        if batch is None:
            tqdm.write('==> Current batch is None')
            continue
        input_data, meta = batch
        img_gr1, img_gr2, label1, img_sa1, img_sa2, label2 = [x.to(device) for x in input_data]
        label1, label2 = label1.float(), label2.float()
        indices, keys_gr, keys_sa = meta['indices'], meta['keys_gr'], meta['keys_sa']
        B = img_gr1.shape[0]
        # NOTE replace the satellite by another one, for debug
        # query_sa[...] = 0
        
        # forward
        output_gr, output_sa = model(img_gr1, img_gr2, img_sa1, img_sa2)
        output_gr, output_sa = output_gr.squeeze(-1), output_sa.squeeze(-1)
        # _, pred_gr = torch.max(output_gr, dim=1)
        # _, pred_sa = torch.max(output_sa, dim=1)
        # NOTE visualize the descriptor for debug
        # if i % 50 == 0:
            # visualize_desc(descQ_gr, descQ_sa)

        # calculate loss, back propagate, update weights
        optimizer.zero_grad()
        loss_gr = criterion(output_gr, label1)
        loss_sa = criterion(output_sa, label2)
        # loss_gr = cycle_mse_loss(output_gr, label1)
        # loss_sa = cycle_mse_loss(output_sa, label2)
        loss = loss_gr + loss_sa
        loss /= B
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        # # NOTE visualize batch and score for debug
        # # visualize_plain_batch_pretrain(batch)
        # Bv = min(B, 6)
        # fig, axes = plt.subplots(nrows=Bv, ncols=4, figsize=(10,10 * Bv / 2))
        # fig.suptitle(f'Navigate dataloader of CVACT: current batch', fontsize=12)
        # fig.tight_layout()
        # fig.subplots_adjust(top=0.9)

        # for i in range(Bv):
        #     axes[i,0].imshow(np.transpose(denormalize(img_gr1[i].detach().cpu().numpy()),(1,2,0)))
        #     axes[i,0].set_title(f"Sample {i} ==> ground image\nidx: {indices[i]}, file name: {keys_gr[i]}, label: {label1[i]}", fontsize=8)
        #     axes[i,1].imshow(np.transpose(denormalize(img_gr2[i].detach().cpu().numpy()),(1,2,0)))
        #     axes[i,2].imshow(np.transpose(denormalize(img_sa1[i].detach().cpu().numpy()),(1,2,0)))
        #     axes[i,2].set_title(f"Sample {i} ==> satellite image\nidx: {indices[i]}, file name: {keys_sa[i]}, label: {label2[i]}", fontsize=8)
        #     axes[i,3].imshow(np.transpose(denormalize(img_sa2[i].detach().cpu().numpy()),(1,2,0)))
        # plt.show()
        # # visualize_scores(output, label)

        batch_loss = loss.item()
        epoch_loss += batch_loss
        correct_gr = ((output_gr - label1).abs() < 0.1).sum().item()
        correct_sa = ((output_sa - label2).abs() < 0.1).sum().item()

        if n_batches <= 10 or iteration % (n_batches // 5) == 0:
            # tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}, Accuracy: {:.4f}, {:.4f}".format(
            #     epoch_num, iteration, n_batches, batch_loss, (pred_gr == label1).sum().item() / B, (pred_sa == label2).sum().item() / B))
            tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch_num, iteration, n_batches, batch_loss))
            tqdm.write("==> Epoch[{}]({}/{}): Accuracy: {:.4f}, {:.4f}".format(epoch_num, iteration, n_batches, correct_gr / B, correct_sa / B))
            writer.add_scalar('Train/Loss', batch_loss, ((epoch_num - 1) * n_batches) + iteration)

    avg_loss = epoch_loss / n_batches
    tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, avg_loss))
    writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)

    # tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
    # tqdm.write('Cached:    ' + humanbytes(torch.cuda.memory_reserved()))