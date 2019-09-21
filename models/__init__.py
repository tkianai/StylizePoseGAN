
import torch
from .posegan_model import PoseGANModel

def build_model(opt):

    if opt.model == 'poseGAN':
        model = PoseGANModel()
    
    else:
        raise NotImplementedError("{} not supported yet!".format(opt.model))

    model.initialize(opt)
    print("Model [{}] was created!".format(model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
