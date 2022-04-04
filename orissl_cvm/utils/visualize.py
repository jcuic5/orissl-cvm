import numpy as np
import matplotlib.pyplot as plt

def denormalize(im):
	image = im.numpy()
	im = (image - np.min(image)) / (np.max(image) - np.min(image))
	im = np.ascontiguousarray(im * 255, dtype=np.uint8)
	return im

def visualize_triplet(batch, sample_idx):

	query, positive, negatives, meta = batch
	negCounts, indices, keys = meta['negCounts'], meta['indices'], meta['keys']

	nc = 0

	num_ns = negCounts[sample_idx].item()
	num_qpns = num_ns + 2

	fig, axes = plt.subplots(nrows=num_qpns, ncols=2, figsize=(15,15))
	fig.suptitle(
		f'Batch sample {sample_idx}: query, positive, and {num_ns} negatives',
		fontsize=15)
	fig.tight_layout()
	fig.subplots_adjust(top=0.95)
	
	axes[0,0].imshow(np.transpose(denormalize(query[0][sample_idx]),(1,2,0)))
	axes[0,0].set_title(
		f"Query ==> ground image\n{keys[sample_idx]['query']['gr_img']}")

	axes[0,1].imshow(np.transpose(denormalize(query[1][sample_idx]),(1,2,0)))
	axes[0,1].set_title(
		f"Query ==> satellite image\n{keys[sample_idx]['query']['sa_img']}")

	axes[1,0].imshow(np.transpose(denormalize(positive[0][sample_idx]),(1,2,0)))
	axes[1,0].set_title(
		f"Positive ==> ground image\n{keys[sample_idx]['positive']['gr_img']}")
	
	axes[1,1].imshow(np.transpose(denormalize(positive[1][sample_idx]),(1,2,0)))
	axes[1,1].set_title(
		f"Positive ==> satellite image\n{keys[sample_idx]['positive']['sa_img']}")

	for i in range(num_ns):
		axes[2+i,0].imshow(np.transpose(denormalize(negatives[0][nc+i]),(1,2,0)))
		axes[2+i,0].set_title(
			f"Negative {i} ==> ground image\n{keys[sample_idx]['negatives'][i]['gr_img']}")

		axes[2+i,1].imshow(np.transpose(denormalize(negatives[1][nc+i]),(1,2,0)))
		axes[2+i,1].set_title(
			f"Negative {i} ==> satellite image\n{keys[sample_idx]['negatives'][i]['sa_img']}")
	nc += num_ns
	plt.show()

def visualize_dataloader(training_loader):

	bs = training_loader.batch_size
	it = iter(training_loader)
	while True:
		try:
			batch = next(it)
			for i in range(bs):
				visualize_triplet(batch, i)
		except StopIteration:
			print("Data loader ran out.")


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