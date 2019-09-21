

import os
from PIL import Image
import lmdb
from io import BytesIO
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
import random


class MultiResolutionPoseDataset(data.Dataset):

    def __init__(self):
        super().__init__()

    def initialize(self, root, training=True, resolution=8):
        self.root = root
        self.training = training
        self.env = lmdb.open(
            self.root,
            readonly=True,
            lock=False,
            meminit=False,
        )
        if not self.env:
            raise IOError("Cannot open lmdb dataset", self.root)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get("Length".encode('utf-8')).decode('utf-8'))

        self.resolution = resolution
        self.max_resolution = 1024
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def name(self):
        return 'MultiResolutionPoseDataset'

    def __getitem__(self, index):

        with self.env.begin(write=False) as txn:
            img_key = "Image-{}-{:0>7d}".format(self.resolution, index).encode('utf-8')
            opose_key = "Openpose-{}-{:0>7d}".format(self.resolution, index).encode('utf-8')
            max_opose_key = "Openpose-{}-{:0>7d}".format(self.max_resolution, index).encode('utf-8')

            img_bytes = txn.get(img_key)
            opose_bytes = txn.get(opose_key)
            max_opose_bytes = txn.get(max_opose_key)

        # Get data
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer).convert('RGB')
        buffer = BytesIO(opose_bytes)
        opose = Image.open(buffer).convert('RGB')
        buffer = BytesIO(max_opose_bytes)
        max_opose = Image.open(buffer).convert('RGB')

        if self.training and random.random < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            opose = opose.transpose(Image.FLIP_LEFT_RIGHT)
            max_opose = max_opose.transpose(Image.FLIP_LEFT_RIGHT)

        opose = self.transforms(opose)
        img = self.transforms(img)
        max_opose = self.transforms(max_opose)

        input_dict = {'label': opose, 'image': img, 'path': "Image-{}-{:0>7d}.png".format(self.resolution, index), 'style': max_opose}

        return input_dict

    def __len__(self):
        return self.length

    def get_resolution(self):
        return self.resolution

    def set_resolution(self, resolution):
        self.resolution = resolution
