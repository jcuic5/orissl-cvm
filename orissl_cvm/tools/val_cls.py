import numpy as np
import torch
import faiss
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from orissl_cvm.datasets.cvact_dataset import ImagePairsFromList
from orissl_cvm.utils import input_transform
from orissl_cvm.tools.visualize import visualize_desc
from orissl_cvm.tools.visualize import visualize_scores, visualize_plain_batch_pretrain


def val_cls(eval_loader, model, device, cfg, writer, epoch_num=0, write_tboard=False, pbar_position=0):
    if device.type == 'cuda':
        cuda = True
    else:
        cuda = False

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        tqdm.write('====> Extracting Features')

        for iteration, batch in enumerate(tqdm(eval_loader, 
                position=pbar_position, leave=False, desc='Test Iter'.rjust(15)), 1):
            if batch is None: 
                tqdm.write('====> Batch data iteration is None. Probably caused by corrupted file')
                continue
            query_gr, query_sa, label, meta = batch
            # query_sa[::2, ...] = 0
            indices, keys = meta['indices'], meta['keys']
            B = query_gr.shape[0]
            query_gr, query_sa, label = query_gr.to(device), query_sa.to(device), label.to(device)

            output = model(query_gr, query_sa)
            _, predicted = torch.max(output, dim=1)
            total += B
            correct += (predicted == label).sum().item()

            if iteration % 1 == 0:
                tqdm.write("====> Batch accuracy: {:.4f}".format((predicted == label).sum().item() / B))

            # NOTE visualize batch and score for debug
            # visualize_plain_batch_pretrain(batch)
            # visualize_scores(output, label)

    acc = correct / total
    tqdm.write("====> Accuracy on validation set: {:.4f}".format(acc))
    if write_tboard:
        writer.add_scalar('Val/Accuracy', acc, epoch_num)

    return acc
