from configparser import Interpolation
import imp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction import image
import torch
import torch.nn.functional as F
from datetime import datetime
from orissl_cvm import PACKAGE_ROOT_DIR
from os.path import join
import sys

def denormalize(im):
	im = (im - np.min(im)) / (np.max(im) - np.min(im))
	im = np.ascontiguousarray(im * 255, dtype=np.uint8)
	return im


def visualize_assets(*assets, mode='image', max_nrows=6, dcn=True, caption='Batch images', **meta):
	B = assets[0].shape[0]
	if mode == 'descriptor': 
		C = assets[0].shape[1]
		vis_ratio = 10
		assets = [F.adaptive_avg_pool1d(x.unsqueeze(1), int(C/vis_ratio)).squeeze(1) for x in assets]
	if mode == 'score':
		C = assets.shape[1]
		assets = F.log_softmax(assets, dim=1)
		assets = F.adaptive_avg_pool1d(assets.unsqueeze(1), int(C/vis_ratio)).squeeze(1)
	nrows = min(B, max_nrows)
	ncols = len(assets)
	assets = [x.detach().cpu().numpy() for x in assets] if dcn else assets

	fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10,10 * nrows / 2))
	fig.suptitle(caption, fontsize=12)
	fig.tight_layout()
	fig.subplots_adjust(top=0.9)

	if mode == 'descriptor':
		random = axes[0,0].imshow(np.random.random((1, int(C/vis_ratio))), cmap='viridis', interpolation='none')
		fig.colorbar(random, ax=axes[0:,0:], location='right', shrink=0.2)

	for i in range(nrows):
		for j in range(ncols):
			if mode == 'image':
				im = assets[j][i]
				if len(im.shape) == 3:
					axes[i,j].imshow(np.transpose(denormalize(im), (1,2,0)))
				elif len(im.shape) == 2:
					axes[i,j].imshow(denormalize(im))
			elif mode == 'descriptor':
				axes[i,j].imshow(assets[j][i:i+1], cmap='viridis', interpolation='none')
				axes[i,j].set_aspect(10)
				axes[i,j].set_title(f"Sample {i} ==> ground descriptor", fontsize=8)
				# axes[i,j].axis('off')
				axes[i,j].get_yaxis().set_visible(False)
			elif mode == 'score':
				axes[i,j].plot(range(assets[i].shape[0]), assets[i], c='b')
			info = f"Sample {i} \n"
			for k, v in meta.items(): 
				info += f"{k}: {v[i]} "
			axes[i,j].set_title(info, fontsize=8)
	plt.show()
	# plt.savefig(join(sys.path[0], 'desc_visualize_for_debug', f"{datetime.now().strftime('%b%d_%H-%M-%S')}.png"))


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


def visualize_dataloader(training_loader):
	bs = training_loader.batch_size
	it = iter(training_loader)
	while True:
		try:
			batch = next(it)
			# visualize
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