import torch
from tqdm.auto import tqdm
from clsslcvm.tools import register_hook, gen_cam, show_cam_on_image
from clsslcvm.tools.visualize import visualize_assets
from clsslcvm.loss import *
import torchvision.transforms.functional as F


def train_epoch(dataloader, model, optimizer, scheduler, criterion, device, epoch_num, cfg, writer):
    batch_size = cfg.train.batch_size
    epoch_loss = 0
    n_batches = (dataloader.train_data_size + batch_size - 1) // batch_size
    if cfg.train.check_align_and_uniform:
        epoch_loss_a = 0
        epoch_loss_u = 0
    model.train()
    if cfg.train.grad_cam: 
        fmp_list_gr, fmp_list_sa, grad_list_gr, grad_list_sa, d_list = register_hook(model)

    for it in tqdm(range(n_batches), position=1, leave=False, desc='Train Iter'.rjust(15)):
        batch_sat, batch_grd, batch_list, delta_list = dataloader.next_batch_train(batch_size)
        B, C, H, W = batch_sat.shape
        # forward
        batch_grd, batch_sat = batch_grd.to(device), batch_sat.to(device)
        desc_grd, desc_sat = model(batch_grd, batch_sat)
        # calculate loss, back propagate, update weights
        optimizer.zero_grad()
        loss = 0
        for i in range(batch_size):
            for j in range(i + 1, batch_size):
                loss += criterion(desc_grd[i: i+1], desc_sat[i: i+1], desc_sat[j: j+1])
                loss += criterion(desc_sat[i: i+1], desc_grd[i: i+1], desc_grd[j: j+1])
        loss /= batch_size
        if cfg.train.grad_cam:
            d_gr, d_sa = d_list[2*it], d_list[2*it+1]
            product = (d_gr * d_sa).sum()
            product.backward()
        else:
            loss.backward()
        optimizer.step()
        if scheduler is not None: 
            scheduler.step()
        # NOTE check alignment & uniformity properties of our positives
        if cfg.train.check_align_and_uniform:
            loss_a = align_loss(desc_grd, desc_sat) / batch_size
            loss_u = uniform_loss(desc_grd) + uniform_loss(desc_sat) / batch_size
        # NOTE check grad-cam
        if cfg.train.grad_cam:
            # fmap_gr = fmp_list_gr[it].cpu().data.numpy().squeeze()
            # grad_gr = grad_list_gr[it].cpu().data.numpy().squeeze()
            # fmap_sa = fmp_list_sa[it].cpu().data.numpy().squeeze()
            # grad_sa = grad_list_sa[it].cpu().data.numpy().squeeze()
            # cam_gr = gen_cam(fmap_gr, grad_gr)
            # cam_sa = gen_cam(fmap_sa, grad_sa)
            visualize_assets(batch_grd, batch_sat)
            # visualize_assets(batch_grd, torch.tensor(fmap_gr).mean(1), batch_sat, torch.tensor(fmap_sa).mean(1))
            # visualize_assets(show_cam_on_image(batch_grd, torch.tensor(fmap_gr).mean(1)), show_cam_on_image(batch_sat, torch.tensor(fmap_sa).mean(1)))
            # visualize_assets(batch_grd, torch.tensor(cam_gr), batch_sat, torch.tensor(cam_sa))
            # visualize_assets(show_cam_on_image(batch_grd, torch.tensor(cam_gr)), show_cam_on_image(batch_sat, torch.tensor(cam_sa)))     
            # visualize_assets(desc_grd, desc_sat, mode='descriptor')
        del batch_grd, batch_sat, desc_grd, desc_sat

        batch_loss = loss.item()
        epoch_loss += batch_loss
        if cfg.train.check_align_and_uniform:
            batch_loss_a = loss_a.item()
            epoch_loss_a += batch_loss_a
            batch_loss_u = loss_u.item()
            epoch_loss_u += batch_loss_u
        if it % (n_batches // 100) == 0 or n_batches <= 10:
            if not cfg.train.check_align_and_uniform:
                tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch_num, it, n_batches, batch_loss))
                writer.add_scalar('Train/Loss', batch_loss, ((epoch_num - 1) * n_batches) + it)
            else: 
                tqdm.write("==> Epoch[{}]({}/{}): Loss: {:.4f}, Align loss: {:.4f}, Uniform loss: {:.4f}".format(epoch_num, it, n_batches, batch_loss, batch_loss_a, batch_loss_u))
                writer.add_scalar('Train/Loss', batch_loss, ((epoch_num - 1) * n_batches) + it)
                writer.add_scalar('Train/Loss_a', batch_loss_a, ((epoch_num - 1) * n_batches) + it)
                writer.add_scalar('Train/Loss_u', batch_loss_u, ((epoch_num - 1) * n_batches) + it)

    avg_loss = epoch_loss / n_batches
    if not cfg.train.check_align_and_uniform:
        tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch_num, avg_loss))
        writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)
    else:
        tqdm.write("===> Epoch {} Complete: Avg. Loss: {:.4f}, Align loss: {:.4f}, Uniform loss: {:.4f}".format(epoch_num, avg_loss, epoch_loss_a / n_batches, epoch_loss_u / n_batches))
        writer.add_scalar('Train/AvgLoss', avg_loss, epoch_num)
        writer.add_scalar('Train/AvgLoss_a', epoch_loss_a / n_batches, epoch_num)
        writer.add_scalar('Train/AvgLoss_u', epoch_loss_u / n_batches, epoch_num)