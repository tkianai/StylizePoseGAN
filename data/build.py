
from . import PoseDataLoader


def build_dataloader(root, label_size, training=True, resolution=8, batch_size=8):
    data_loader = PoseDataLoader()
    print(data_loader.name())
    data_loader.initialize(root, label_size, training=training,
                           resolution=resolution, batch_size=batch_size)

    return data_loader

