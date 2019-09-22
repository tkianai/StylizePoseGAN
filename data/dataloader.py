

from .dataset import MultiResolutionPoseDataset
import torch.utils.data


def build_dataset(root, label_size, training=True, resolution=8):

    dataset = MultiResolutionPoseDataset()
    print("[{}] was created!".format(dataset.name()))
    dataset.initialize(root, label_size, training, resolution)

    return dataset


class PoseDataLoader():
    def __init__(self):
        pass

    def initialize(self, root, label_size, training=True, resolution=8, batch_size=8):
        self.dataset = build_dataset(root, label_size, training, resolution)
        self.batch_size = batch_size
        self.update(resolution, batch_size)


    def update(self, resolution=None, batch_size=None):
        if resolution is not None:
            self.dataset.set_resolution(resolution)
        if batch_size is not None:
            self.batch_size = batch_size
        
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.dataset.training,
            drop_last=True,
            num_workers=16,
        )

    def name(self):
        return "PoseDataLoader"

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
