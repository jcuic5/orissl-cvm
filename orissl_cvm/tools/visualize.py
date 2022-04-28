from configparser import Interpolation
import imp
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from datetime import datetime
from orissl_cvm import PACKAGE_ROOT_DIR
from os.path import join
import sys

def denormalize(im):
	image = im.numpy()
	im = (image - np.min(image)) / (np.max(image) - np.min(image))
	im = np.ascontiguousarray(im * 255, dtype=np.uint8)
	return im

def visualize_triplet(batch, sample_idx):

	query, negatives, meta = batch
	negCounts, indices, keys = meta['negCounts'], meta['indices'], meta['keys']

	B = query[0].shape[0]

	num_ns = negCounts[sample_idx].item()
	num_qns = num_ns + 1

	neg_start = 0
	start = 0
	if sample_idx > 0: 
		neg_start = negCounts[:sample_idx].sum().item()
		start = neg_start + sample_idx * 1
	# print(sample_idx, start)

	fig, axes = plt.subplots(nrows=num_qns, ncols=2, figsize=(15,9))
	fig.suptitle(
		f'Navigate dataloader of CVACT: current batch, sample {sample_idx} (1 query and {num_ns} negatives)',
		fontsize=15)
	fig.tight_layout()
	fig.subplots_adjust(top=0.9)
	
	axes[0,0].imshow(np.transpose(denormalize(query[0][sample_idx]),(1,2,0)))
	axes[0,0].set_title(
		f"Query ==> ground image\nidx: {indices[start]}, file name: {keys[sample_idx]['query']['img_gr']}")

	axes[0,1].imshow(np.transpose(denormalize(query[1][sample_idx]),(1,2,0)))
	axes[0,1].set_title(
		f"Query ==> satellite image\nidx: {indices[start]}, file name: {keys[sample_idx]['query']['img_sa']}")

	# axes[1,0].imshow(np.transpose(denormalize(positive[0][sample_idx]),(1,2,0)))
	# axes[1,0].set_title(
	# 	f"Positive ==> ground image\n{keys[sample_idx]['positive']['img_gr']}")
	
	# axes[1,1].imshow(np.transpose(denormalize(positive[1][sample_idx]),(1,2,0)))
	# axes[1,1].set_title(
	# 	f"Positive ==> satellite image\n{keys[sample_idx]['positive']['img_sa']}")

	for i in range(num_ns):
		axes[1+i,0].imshow(np.transpose(denormalize(negatives[0][neg_start+i]),(1,2,0)))
		axes[1+i,0].set_title(
			f"Negative {i} ==> ground image\nidx: {indices[start+i+1]}, file name: {keys[sample_idx]['negatives'][i]['img_gr']}")

		axes[1+i,1].imshow(np.transpose(denormalize(negatives[1][neg_start+i]),(1,2,0)))
		axes[1+i,1].set_title(
			f"Negative {i} ==> satellite image\nidx: {indices[start+i+1]}, file name: {keys[sample_idx]['negatives'][i]['img_sa']}")

	plt.show()

def visualize_plain_batch(batch):

	query, meta = batch
	indices, keys, qpn_mat = meta['indices'], meta['keys'], meta['qpn_mat']
	B = query[0].shape[0]

	fig, axes = plt.subplots(nrows=B, ncols=2, figsize=(10,10*B/2))
	fig.suptitle(f'Navigate dataloader of CVACT: current batch', fontsize=12)
	fig.tight_layout()
	fig.subplots_adjust(top=0.9)
	print(qpn_mat)

	for i in range(B):
		axes[i,0].imshow(np.transpose(denormalize(query[0][i]),(1,2,0)))
		axes[i,0].set_title(
			f"Sample {i} ==> ground image\nidx: {indices[i]}, file name: {keys[i]['img_gr']}", fontsize=8)

		axes[i,1].imshow(np.transpose(denormalize(query[1][i]),(1,2,0)))
		axes[i,1].set_title(
			f"Sample {i} ==> satellite image\nidx: {indices[i]}, file name: {keys[i]['img_sa']}", fontsize=8)

	plt.show()

def visualize_plain_batch_pretrain(batch):

	query_gr, query_sa, label, meta = batch
	indices, keys = meta['indices'], meta['keys']
	B = query_gr.shape[0]
	Bv = min(B, 6)

	fig, axes = plt.subplots(nrows=Bv, ncols=2, figsize=(10,10 * Bv / 2))
	fig.suptitle(f'Navigate dataloader of CVACT: current batch', fontsize=12)
	fig.tight_layout()
	fig.subplots_adjust(top=0.9)

	for i in range(Bv):
		axes[i,0].imshow(np.transpose(denormalize(query_gr[i]),(1,2,0)))
		axes[i,0].set_title(
			f"Sample {i} ==> ground image\nidx: {indices[i]}, file name: {keys[i]['img_gr']}, label: {label[i]}", fontsize=8)

		axes[i,1].imshow(np.transpose(denormalize(query_sa[i]),(1,2,0)))
		axes[i,1].set_title(
			f"Sample {i} ==> satellite image\nidx: {indices[i]}, file name: {keys[i]['img_sa']}", fontsize=8)

	plt.show()

def visualize_dataloader(training_loader):
	bs = training_loader.batch_size
	it = iter(training_loader)
	while True:
		try:
			batch = next(it)
			# for i in range(bs):
			# 	visualize_triplet(batch, i)
			# visualize_plain_batch(batch)
			visualize_plain_batch_pretrain(batch)
		except StopIteration:
			print("Data loader ran out.")
			break

def visualize_dataloader_interact(training_loader):
	'''Note: To be launched in Jupyter'''
	import ipywidgets as widgets
	from IPython.display import display, clear_output

	button = widgets.Button(
		description='Next Batch',
		layout=widgets.Layout(width='10%')
	)
	out = widgets.Output()
	bs = training_loader.batch_size
	it = iter(training_loader)

	def on_button_clicked(_):
		#https://medium.com/@jdchipox/how-to-interact-with-jupyter-33a98686f24e
		with out:
			try:
				batch = next(it)
			except StopIteration:
				print("Data loader ran out.")
			clear_output()
			# display(f'')
			sample_slider = widgets.IntSlider(
				value=0, min=0, max=bs-1, step=1, 
				description='Sample:',
				layout=widgets.Layout(width='25%')
			)
			widgets.interact(lambda sample_idx: visualize_triplet(batch, sample_idx),
							sample_idx=sample_slider)
	button.on_click(on_button_clicked)
	# displaying button and its output together
	widgets.VBox([button,out])

def visualize_desc(desc_gr, desc_sa, vis_ratio=10):
	B = desc_gr.shape[0]
	C = desc_gr.shape[1]
	desc_gr = F.adaptive_avg_pool1d(desc_gr.unsqueeze(1), int(C/vis_ratio)).squeeze(1)
	desc_sa = F.adaptive_avg_pool1d(desc_sa.unsqueeze(1), int(C/vis_ratio)).squeeze(1)

	desc_gr_cdn, desc_sa_cdn = desc_gr.detach().cpu().numpy(), desc_sa.detach().cpu().numpy()

	fig, axes = plt.subplots(nrows=B, ncols=2, figsize=(15,15))
	fig.suptitle(f'Output descriptors of current batch', fontsize=12)
	fig.tight_layout()
	fig.subplots_adjust(top=0.9)

	random = axes[0,0].imshow(np.random.random((1, int(C/vis_ratio))), cmap='viridis', interpolation='none')
	fig.colorbar(random, ax=axes[0:,0:], location='right', shrink=0.2)

	for i in range(B):
		im0 = axes[i,0].imshow(desc_gr_cdn[i:i+1], cmap='viridis', interpolation='none')
		axes[i,0].set_aspect(25)
		axes[i,0].set_title(f"Sample {i} ==> ground descriptor", fontsize=8)
		# axes[i,0].axis('off')
		axes[i,0].get_yaxis().set_visible(False)

		im1 = axes[i,1].imshow(desc_sa_cdn[i:i+1], cmap='viridis', interpolation='none')
		axes[i,1].set_aspect(25)
		axes[i,1].set_title(f"Sample {i} ==> satellite descriptor", fontsize=8)
		# axes[i,1].axis('off')
		axes[i,1].get_yaxis().set_visible(False)

	# plt.show()
	plt.savefig(join(sys.path[0], 'desc_visualize_for_debug', f"{datetime.now().strftime('%b%d_%H-%M-%S')}.png"))

def visualize_scores(scores, label, vis_ratio=1, mode='plot'):
	logits = F.log_softmax(scores, dim=1)
	B = scores.shape[0]
	C = scores.shape[1]
	Bv = min(B, 6)
	logits = F.adaptive_avg_pool1d(logits.unsqueeze(1), int(C/vis_ratio)).squeeze(1)

	logits_cdn = logits.detach().cpu().numpy()

	fig, axes = plt.subplots(nrows=Bv, ncols=1, figsize=(5,5))
	fig.suptitle(f'Output scores (after log_softmax) of current batch', fontsize=12)
	fig.tight_layout()
	fig.subplots_adjust(top=0.9)

	if mode == 'plot':
		for i in range(Bv):
			axes[i].plot(range(logits_cdn[i].shape[0]), logits_cdn[i], c='b')
			axes[i].set_title(f"Sample {i} ==> scores. The gt label is: {label[i]}", fontsize=8)

	elif mode == 'cmap':
		random = axes[0].imshow(np.random.random((1, int(C/vis_ratio))), cmap='viridis', interpolation='none')
		fig.colorbar(random, ax=axes[0:], location='right', shrink=0.2)

		for i in range(Bv):
			im0 = axes[i].imshow(logits_cdn[i:i+1], cmap='viridis', interpolation='none')
			axes[i].set_aspect(25)
			axes[i].set_title(f"Sample {i} ==> scores. The gt label is: {label[i]}", fontsize=8)
			axes[i,0].axis('off')
			axes[i].get_yaxis().set_visible(False)

	plt.show()


def visualize_cl(batch):
    (image1, image2), indices = batch
    # image1_cdn, image2_cdn = image1.numpy(), image2.numpy()

    B = image1.shape[0]
    B_vis = min(B, 6)
    
    fig, axes = plt.subplots(nrows=B_vis, ncols=2, figsize=(10,10*B_vis/2))
    fig.suptitle(f'Navigate dataloader of CVACT: current batch', fontsize=12)
    fig.tight_layout()
    fig.subplots_adjust(top=0.9)
    
    for i in range(B_vis):
        axes[i,0].imshow(np.transpose(denormalize(image1[i]),(1,2,0)))
        axes[i,0].set_title(
			f"Sample {i} ==> ground image 1\nidx: {indices[i]}", fontsize=8)
        axes[i,1].imshow(np.transpose(denormalize(image2[i]),(1,2,0)))
        axes[i,1].set_title(
			f"Sample {i} ==> ground image 2\nidx: {indices[i]}", fontsize=8)
            
    plt.show()