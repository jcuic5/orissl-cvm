import torch
from tqdm.auto import trange, tqdm
from torch.utils.data import DataLoader
from orissl_cvm.tools import humanbytes
from orissl_cvm.datasets.cvact_dataset import CVACTDataset
from orissl_cvm.tools import forward_hook, backward_hook, gen_cam, show_cam_on_image
from orissl_cvm.tools.visualize import visualize_assets
from orissl_cvm.loss import *
import torchvision.transforms.functional as F
from torchvision.transforms import InterpolationMode


def train_epoch(train_dataset, training_data_loader, model, 
                optimizer, scheduler, criterion, device, 
                epoch_num, cfg, writer):
    epoch_loss = 0
    n_batches = (len(train_dataset.qIdx) + cfg.train.batch_size - 1) // cfg.train.batch_size

    if cfg.train.check_align_and_uniform:
        epoch_loss_a = 0
        epoch_loss_u = 0

    model.train()
    local_progress = tqdm(training_data_loader, position=1, leave=False, desc='Train Iter'.rjust(15))

    if cfg.train.grad_cam:
        if not cfg.model.shared:
            fmp_list_gr, fmp_list_sa = [], []
            grad_list_gr, grad_list_sa = [], []
            d_list = []
            fh_gr = lambda module, input, output : forward_hook(module, input, output, fmp_list=fmp_list_gr)
            bh_gr = lambda module, grad_in, grad_out : backward_hook(module, grad_in, grad_out, grad_list=grad_list_gr)
            fh_sa = lambda module, input, output : forward_hook(module, input, output, fmp_list=fmp_list_sa)
            bh_sa = lambda module, grad_in, grad_out : backward_hook(module, grad_in, grad_out, grad_list=grad_list_sa)
            fh_d = lambda module, input, output : forward_hook(module, input, output, fmp_list=d_list)
            model.features_gr[-1].register_forward_hook(fh_gr)
            model.features_gr[-1].register_full_backward_hook(bh_gr)
            model.features_sa[-1].register_forward_hook(fh_sa)
            model.features_sa[-1].register_full_backward_hook(bh_sa)
            model.pool[-2].register_forward_hook(fh_d)
        else:
            fmp_list = []
            grad_list = []
            fh = lambda module, input, output : forward_hook(module, input, output, fmp_list=fmp_list)
            bh = lambda module, grad_in, grad_out : backward_hook(module, grad_in, grad_out, grad_list=grad_list)
            model.features[-1].register_forward_hook(fh)
            model.features[-1].register_full_backward_hook(bh)

    for iteration, batch in enumerate(local_progress):
        # prepare data
        if batch is None:
            tqdm.write('==> Current batch is None')
            continue
        input_data, meta = batch
        query_gr, query_sa = input_data
        indices, keys_gr, keys_sa = meta['indices'], meta['keys_gr'], meta['keys_sa']
        B, C, H, W = query_gr.shape
        qpn_mat = torch.from_numpy(train_dataset.qpn_matrix[indices, :][:, indices]).to(device)
        qn_triplets = torch.nonzero(qpn_mat == 0, as_tuple=False)
        n_triplets = qn_triplets.shape[0]
        if n_triplets == 0:
            del query_gr, query_sa
            continue
        
        # forward
        query_gr, query_sa = query_gr.to(device), query_sa.to(device)
        # NOTE for debug
        # visualize_assets(query_gr, query_sa)
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
        # loss += uniform_loss(descQ_gr) + uniform_loss(descQ_sa)
        loss /= n_triplets # normalise by actual number of negatives
        if cfg.train.grad_cam:
            d_gr, d_sa = d_list[2*iteration], d_list[2*iteration+1]
            product = (d_gr * d_sa).sum()
            product.backward()
        else:
            loss.backward()

        # NOTE check alignment & uniformity properties of our positives
        if cfg.train.check_align_and_uniform:
            loss_a = align_loss(descQ_gr, descQ_sa)
            loss_u = uniform_loss(descQ_gr) + uniform_loss(descQ_sa)
            loss_a /= n_triplets
            loss_u /= n_triplets

        # NOTE check grad-cam
        if cfg.train.grad_cam:
            if not cfg.model.shared:
                fmap_gr = fmp_list_gr[iteration].cpu().data.numpy().squeeze()
                grad_gr = grad_list_gr[iteration].cpu().data.numpy().squeeze()
                fmap_sa = fmp_list_sa[iteration].cpu().data.numpy().squeeze()
                grad_sa = grad_list_sa[iteration].cpu().data.numpy().squeeze()
            else:
                fmap_gr = fmp_list[2*iteration].cpu().data.numpy().squeeze()
                grad_gr = grad_list[2*iteration].cpu().data.numpy().squeeze()
                fmap_sa = fmp_list[2*iteration+1].cpu().data.numpy().squeeze()
                grad_sa = grad_list[2*iteration+1].cpu().data.numpy().squeeze()
            cam_gr = gen_cam(fmap_gr, grad_gr)
            cam_sa = gen_cam(fmap_sa, grad_sa)
            # visualize_assets(query_gr, query_sa)
            visualize_assets(query_gr, torch.tensor(fmap_gr).mean(1), query_sa, torch.tensor(fmap_sa).mean(1))
            visualize_assets(show_cam_on_image(query_gr, torch.tensor(fmap_gr).mean(1)), show_cam_on_image(query_sa, torch.tensor(fmap_sa).mean(1)))
            visualize_assets(query_gr, torch.tensor(cam_gr), query_sa, torch.tensor(cam_sa))
            visualize_assets(show_cam_on_image(query_gr, torch.tensor(cam_gr)), show_cam_on_image(query_sa, torch.tensor(cam_sa)))     
            visualize_assets(descQ_gr, descQ_sa, mode='descriptor')

        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        del query_gr, query_sa, descQ_gr, descQ_sa

        batch_loss = loss.item()
        epoch_loss += batch_loss
        if cfg.train.check_align_and_uniform:
            batch_loss_a = loss_a.item()
            epoch_loss_a += batch_loss_a
            batch_loss_u = loss_u.item()
            epoch_loss_u += batch_loss_u
        if iteration % (n_batches // 5) == 0 or n_batches <= 10:
            if not cfg.train.check_align_and_uniform:
                tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch_num, iteration, n_batches, batch_loss))
                writer.add_scalar('Train/Loss', batch_loss, ((epoch_num - 1) * n_batches) + iteration)
            else: 
                tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}, Align loss: {:.4f}, Uniform loss: {:.4f}".format(epoch_num, iteration, n_batches, batch_loss, batch_loss_a, batch_loss_u))
                writer.add_scalar('Train/Loss', batch_loss, ((epoch_num - 1) * n_batches) + iteration)
                writer.add_scalar('Train/Loss_a', batch_loss_a, ((epoch_num - 1) * n_batches) + iteration)
                writer.add_scalar('Train/Loss_u', batch_loss_u, ((epoch_num - 1) * n_batches) + iteration)

    avg_loss = epoch_loss / n_batches
    if not cfg.train.check_align_and_uniform:
        tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, avg_loss))
        writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)
    else:
        tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}, Align loss: {:.4f}, Uniform loss: {:.4f}".format(epoch_num, avg_loss, epoch_loss_a / n_batches, epoch_loss_u / n_batches))
        writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)
        writer.add_scalar('Train/AvgLoss_a', epoch_loss_a / n_batches, epoch_num)
        writer.add_scalar('Train/AvgLoss_u', epoch_loss_u / n_batches, epoch_num)

    # tqdm.write('Allocated: ' + humanbytes(torch.cuda.memory_allocated()))
    # tqdm.write('Cached:    ' + humanbytes(torch.cuda.memory_reserved()))