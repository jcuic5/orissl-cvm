import imp
import numpy as np
import torch
# import faiss
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from clsslcvm.augmentations import input_transform
from clsslcvm.tools.visualize import visualize_assets, visualize_featdist
from clsslcvm.loss import *


def validate_vigor(grd_descriptor, sat_descriptor, epoch_num, val_set, logger, writer, write_tboard):
    logger.info('===> Compute accuracy')
    accuracy = 0.0
    accuracy_top1 = 0.0
    accuracy_top5 = 0.0
    accuracy_hit = 0.0

    data_amount = 0.0
    dist_array = 2 - 2 * np.matmul(grd_descriptor, np.transpose(sat_descriptor))

    top1_percent = int(dist_array.shape[1] * 0.01) + 1
    top1 = 1
    top5 = 5
    for i in range(dist_array.shape[0]):
        gt_dist = dist_array[i, val_set.grd_sat_label[i][0]]
        prediction = np.sum(dist_array[i, :] < gt_dist)

        dist_temp = np.ones(dist_array[i, :].shape[0])
        dist_temp[val_set.grd_sat_label[i][1:]] = 0
        prediction_hit = np.sum((dist_array[i, :] < gt_dist) * dist_temp)

        if prediction < top1_percent:
            accuracy += 1.0
        if prediction < top1:
            accuracy_top1 += 1.0
        if prediction < top5:
            accuracy_top5 += 1.0
        if prediction_hit < top1:
            accuracy_hit += 1.0
        data_amount += 1.0

    accuracy /= data_amount
    accuracy_top1 /= data_amount
    accuracy_top5 /= data_amount
    accuracy_hit /= data_amount

    logger.info('===> Evaluation epoch %d: accuracy = %.1f%% , top1: %.1f%%, top5: %.1f%%, hit_rate: %.1f%%' % (
                epoch_num, accuracy * 100.0, accuracy_top1 * 100.0, accuracy_top5 * 100.0, accuracy_hit * 100.0))
    all_recalls = {
        'Recall@1%': accuracy,
        'Recall@1': accuracy_top1,
        'Recall@5': accuracy_top5,
        'Hitrate': accuracy_hit
    }  # make dict for output
    if write_tboard:
        writer.add_scalar('Val/Recall@1%', accuracy, epoch_num)
        writer.add_scalar('Val/Recall@1', accuracy_top1, epoch_num)
        writer.add_scalar('Val/Recall@5', accuracy_top5, epoch_num)
        writer.add_scalar('Val/Hitrate', accuracy_hit, epoch_num)

    return all_recalls


# def validate_faiss(val_set, dim, desc_grd_all, desc_sat_all, epoch_num, logger, writer, write_tboard):
#     # start evaluation
#     logger.info('===> Calculating recall @ N')
#     n_values = [1, 5, 100, int(0.01*val_set.sat_list_size)]
#     # for each query, get its positive samples within the threshold distance
#     gt = val_set.grd_sat_label

#     faiss_index = faiss.IndexFlatL2(dim)
#     faiss_index.add(desc_sat_all)
#     _, predictions = faiss_index.search(desc_grd_all, max(n_values))
#     correct_at_n = np.zeros(len(n_values))
#     for qIx, pred in enumerate(predictions):
#         for i, n in enumerate(n_values):
#             # if in top N then also in top NN, where NN > N
#             if np.any(np.in1d(pred[:n], gt[qIx])):
#                 correct_at_n[i:] += 1
#                 break
#     # 简单来说，就是在以这个严格标准下评判（只要pred中的前这些个里面有一个真的是gt里的
#     # 一个就可以）总共有多少比例的qidx
#     recall_at_n = correct_at_n / val_set.grd_list_size

#     all_recalls = {}  # make dict for output
#     for i, n in enumerate(n_values):
#         all_recalls['Recall@' + str(n)] = recall_at_n[i]
#         if write_tboard:
#             writer.add_scalar('Val/Recall@' + str(n), recall_at_n[i], epoch_num)
#     logger.info(all_recalls)
    
#     return gt, predictions, recall_at_n, all_recalls


def val(val_set, val_loader_queries_grd, val_loader_queries_sat, model, device, cfg, logger, writer, 
        epoch_num=0, write_tboard=False, pbar_position=0):
    model.eval()
    dim = val_set.dim
    batch_size = cfg.train.batch_size
    # start extracting features
    with torch.no_grad():
        logger.info('===> Extracting Features')
        desc_grd_all = np.empty((val_set.grd_list_size, dim), dtype=np.float32)
        desc_sat_all = np.empty((val_set.sat_list_size, dim), dtype=np.float32)
        
        local_progress = tqdm(val_loader_queries_grd, position=pbar_position, leave=False, desc='Test Iter (grd)'.rjust(15))
        logger.info('===> Extracting Features for val ground queries')
        n_batches = (val_set.grd_list_size + batch_size - 1) // batch_size
        for it, batch in enumerate(local_progress, 1):
            batch_grd, indices = batch
            #! CVMModelv2
            desc_grd = model.pool(model.features_gr(batch_grd.to(device)))
            desc_grd_all[indices.detach().numpy(), :] = desc_grd.detach().cpu().numpy()
            del batch_grd, desc_grd
            if it % (n_batches // 5) == 0 or n_batches <= 10:
                logger.info("Epoch[{}]({}/{}): indices {} to {} extracted".format(epoch_num, it, n_batches, indices[0], indices[-1]))
        
        local_progress = tqdm(val_loader_queries_sat, position=pbar_position, leave=False, desc='Test Iter (sat)'.rjust(15))
        logger.info('===> Extracting Features for val satellite queries')
        n_batches = (val_set.sat_list_size + batch_size - 1) // batch_size
        for it, batch in enumerate(local_progress, 1):
            batch_sat, indices = batch
            #! CVMModelv2
            desc_sat = model.pool(model.features_sa(batch_sat.to(device)))
            desc_sat_all[indices.detach().numpy(), :] = desc_sat.detach().cpu().numpy()
            del batch_sat, desc_sat
            if it % (n_batches // 5) == 0 or n_batches <= 10:
                logger.info("Epoch[{}]({}/{}): indices {} to {} extracted".format(epoch_num, it, n_batches, indices[0], indices[-1]))

    # gt, predictions, recall_at_n, all_recalls = \
    #         validate_faiss(val_set, dim, desc_grd_all, desc_sat_all, epoch_num, logger, writer, write_tboard)

    all_recalls = validate_vigor(desc_grd_all, desc_sat_all, epoch_num, val_set, logger, writer, write_tboard)

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
    return all_recalls, desc_grd_all, desc_sat_all