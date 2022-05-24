import imp
import numpy as np
import torch
import faiss
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from clsslcvm.datasets.generic_dataset import ImagePairsFromList
from clsslcvm.augmentations import input_transform
from clsslcvm.tools.visualize import visualize_assets, visualize_featdist
from clsslcvm.loss import *


def val(val_dataset, val_dataset_queries, val_dataloader_queries, model, device, cfg, writer, epoch_num=0, write_tboard=False, pbar_position=0):
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

    # start evaluation
    tqdm.write('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20, 50, 100]
    # for each query, get its positive samples within the threshold distance
    gt = val_dataset.all_pos_indices

    faiss_index = faiss.IndexFlatL2(desc_dim)
    faiss_index.add(qFeat_sa)
    _, predictions = faiss_index.search(qFeat_gr, max(n_values))
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    # 简单来说，就是在以这个严格标准下评判（只要pred中的前这些个里面有一个真的是gt里的
    # 一个就可以）总共有多少比例的qidx
    recall_at_n = correct_at_n / len(val_dataset.q_idx)

    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        tqdm.write("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard:
            writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch_num)

    # NOTE for debug visualize desciptor dsitribution using t-SNE
    ul_gr, ul_sa = uniform_loss(torch.tensor(qFeat_gr)).item(), uniform_loss(torch.tensor(qFeat_sa)).item()
    caption = f'\
        Model: {cfg.identifier}\n\
        Unifomity: {ul_gr:.2f}, {ul_sa:.2f}.Recall@1: {recall_at_n[0]:.2f}, Recall@5: {recall_at_n[1]:.2f}'
    visualize_featdist(qFeat_gr, qFeat_sa, caption)

    # # NOTE visualize the retrieval result
    # num_samples = 4
    # recall_at = 3
    # i = 0
    # pred = predictions[i:i+num_samples, :recall_at]
    # g = gt[i:i+num_samples]
    # img_query = torch.cat([val_dataset[x][0][0][None, ...] for x in range(i, i+num_samples)], 0)
    # img_gt = torch.cat([val_dataset[x[0]][0][1][None, ...] for x in g], 0)
    # img_pred = [torch.cat([val_dataset[x][0][1][None, ...] for x in pred[:, y]], 0) for y in range(recall_at)]
    # visualize_assets(img_query, img_gt, *img_pred, mode='image', caption='Retrieval result')
    return all_recalls