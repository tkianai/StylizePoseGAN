
from . import PoseDataLoader


def build_dataloader(root, training=True, resolution=8, batch_size=8):
    data_loader = PoseDataLoader()
    print(data_loader.name())
    data_loader.initialize(root, training=training, resolution=resolution, batch_size=batch_size)

    return data_loader

