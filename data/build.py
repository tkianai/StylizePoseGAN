
from . import PoseDataLoader

def build_dataloader(opt, resolution=8):
    data_loader = PoseDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt, resolution)

    return data_loader

