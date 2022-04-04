import sys
from pathlib import Path
from orissl_cvm.datasets.cvact_dataset import CVACTDataset
from orissl_cvm.utils.tools import input_transform
from orissl_cvm.utils.visualize import visualize_triplet, visualize_dataloader
from torch.utils.data import DataLoader

print(sys.path)

def main():

    root_dir = Path('./data/CVACT_full/').absolute()

    # get transform
    transform = input_transform(resize=(112, 616))

    train_dataset = CVACTDataset(root_dir, nNeg=5, transform=transform, mode='train', 
                            posDistThr=15, negDistThr=100, cached_queries=1000, 
                            cached_negatives=1000, positive_sampling=True, bs=4, threads=8, margin=0.1)

    # divides dataset into smaller cache sets
    train_dataset.new_epoch()

    # creates triplets on the smaller cache set
    train_dataset.update_subcache()

    # create data loader
    opt = {'batch_size': 4, 'shuffle': False, 'collate_fn': CVACTDataset.collate_fn}
    training_loader = DataLoader(train_dataset, **opt)

    # visualize a triplet
    visualize_dataloader(training_loader)


if __name__ == "__main__":
    main()