
import torch
import numpy as np
from PIL import Image
import os


def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    # image_tensor: channel x height x width
    # range: [-1, 1] -> normalize
    # range: [0, 1]  -> non-normalize
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0

    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
        image_numpy = image_numpy[:, :, 0]
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def save_images(visuals, name, save_dir):

    for label, image_numpy in visuals.items():
        image_name = '{}_{}.jpg'.format(name, label)
        save_path = os.path.join(save_dir, image_name)
        util.save_image(image_numpy, save_path)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

