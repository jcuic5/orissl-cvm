import numpy as np
import torch
import torch.nn as nn
import faiss
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from orissl_cvm.datasets.cvact_dataset import ImagePairsFromList


def val(eval_loader, model, device, cfg, writer, epoch_num=0, write_tboard=False, pbar_position=0):
    model.eval()
    correct_gr = 0
    correct_sa = 0
    tloss = 0
    total = 0
    criterion = nn.MSELoss().to(device)
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        local_progress = tqdm(eval_loader, position=pbar_position, leave=False, desc='Test Iter'.rjust(15))
        for iteration, batch in enumerate(local_progress, 1):
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
            total += B
            # correct_gr += (pred_gr == label1).sum().item()
            # correct_sa += (pred_sa == label2).sum().item()

            loss_gr = criterion(output_gr, label1)
            loss_sa = criterion(output_sa, label2)
            loss = loss_gr + loss_sa
            loss /= B
            tloss += loss

    # acc_gr = correct_gr / total
    # acc_sa = correct_sa / total
    tloss /= total
    # tqdm.write("====> Accuracy on validation set: {:.4f}, {:.4f}".format(acc_gr, acc_sa))
    tqdm.write("====> Loss on validation set: {:.4f}".format(tloss))
    if write_tboard:
        # writer.add_scalar('Val/Accuracy_gr', acc_gr, epoch_num)
        # writer.add_scalar('Val/Accuracy_sa', acc_sa, epoch_num)
        writer.add_scalar('Val/Loss', tloss, epoch_num)

    # return (acc_gr + acc_sa) / 2
    return - tloss