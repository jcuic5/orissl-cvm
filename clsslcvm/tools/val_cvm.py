import imp
import numpy as np
import torch
import faiss
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from clsslcvm.augmentations import input_transform
from clsslcvm.tools.visualize import visualize_assets, visualize_featdist
from clsslcvm.loss import *


def val(dataloader, model, device, cfg, writer, epoch_num=0, write_tboard=False, pbar_position=0):
    model.eval()
    batch_size = cfg.train.batch_size
    n_batches_grd = (dataloader.train_data_size + batch_size - 1) // batch_size
    n_batches_sat = (dataloader.test_data_size + batch_size - 1) // batch_size
    dim = dataloader.dim
    # start extracting features
    with torch.no_grad():
        tqdm.write('====> Extracting Features')
        desc_grd_all = np.empty((dataloader.test_sat_data_size, dim), dtype=np.float32)
        desc_sat_all = np.empty((dataloader.test_data_size, dim), dtype=np.float32)
        it = 0
        for i in tqdm(range(n_batches_grd), position=pbar_position, leave=False, desc='Test Iter'.rjust(15)):
            batch_grd = dataloader.next_batch_test_grd(batch_size)
            desc_grd = model.pool_gr(model.features_gr(batch_grd.to(device)))
            desc_grd_all[it: it + desc_grd.shape[0], :] = desc_grd.detach().cpu().numpy()
            it += desc_grd.shape[0]
        for it in tqdm(range(n_batches_sat), position=pbar_position, leave=False, desc='Test Iter'.rjust(15)):
            batch_sat = dataloader.next_batch_test_sat(batch_size)
            desc_sat = model.pool_sa(model.features_sa(batch_sat.to(device)))
            desc_sat_all[it: it + desc_sat.shape[0], :] = desc_grd.detach().cpu().numpy()
            it += desc_grd.shape[0]

    # start evaluation
    tqdm.write('====> Calculating recall @ N')
    n_values = [1, 5, 10, 20, 50, 100]
    # for each query, get its positive samples within the threshold distance
    gt = dataloader.test_label

    faiss_index = faiss.IndexFlatL2(dim)
    faiss_index.add(desc_sat_all)
    _, predictions = faiss_index.search(desc_grd_all, max(n_values))
    correct_at_n = np.zeros(len(n_values))
    for qIx, pred in enumerate(predictions):
        for i, n in enumerate(n_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[qIx])):
                correct_at_n[i:] += 1
                break
    # 简单来说，就是在以这个严格标准下评判（只要pred中的前这些个里面有一个真的是gt里的
    # 一个就可以）总共有多少比例的qidx
    recall_at_n = correct_at_n / dataloader.train_data_size

    all_recalls = {}  # make dict for output
    for i, n in enumerate(n_values):
        all_recalls[n] = recall_at_n[i]
        tqdm.write("====> Recall@{}: {:.4f}".format(n, recall_at_n[i]))
        if write_tboard:
            writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch_num)

    # # NOTE for debug visualize desciptor dsitribution using t-SNE
    # ul_gr, ul_sa = uniform_loss(torch.tensor(desc_grd_all)).item(), uniform_loss(torch.tensor(desc_sat_all)).item()
    # caption = f'\
    #     Model: {cfg.identifier}\n\
    #     Unifomity: {ul_gr:.2f}, {ul_sa:.2f}.Recall@1: {recall_at_n[0]:.2f}, Recall@5: {recall_at_n[1]:.2f}'
    # visualize_featdist(desc_grd_all, desc_sat_all, caption)

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