

from .dataset import MultiResolutionPoseDataset
import torch.utils.data


def build_dataset(opt):

    dataset = MultiResolutionPoseDataset()
    print("[{}] was created!".format(dataset.name()))
    dataset.initialize(opt)

    return dataset


class PoseDataLoader():
    def __init__(self):
        pass

    def initialize(self, opt, resolution=8):
        self.opt = opt
        self.dataset = build_dataset(opt)
        self.dataset.set_resolution(resolution)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.num_workers)
        )

    def name(self):
        return "PoseDataLoader"

    def load_data():

        return self.dataloader

    def __len__(self):
        return len(self.dataset)
