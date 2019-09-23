

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

    def initialize(self, root, label_size, training=True, resolution=8):
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
        self.label_resolution = label_size
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
            label_opose_key = "Openpose-{}-{:0>7d}".format(
                self.label_resolution, index).encode('utf-8')
            imgname_key = "Imgname-{:0>7d}".format(index).encode('utf-8')

            img_bytes = txn.get(img_key)
            opose_bytes = txn.get(opose_key)
            label_opose_bytes = txn.get(label_opose_key)
            imgname = txn.get(imgname_key)

        # Get data
        buffer = BytesIO(img_bytes)
        img = Image.open(buffer).convert('RGB')
        buffer = BytesIO(opose_bytes)
        opose = Image.open(buffer).convert('RGB')
        buffer = BytesIO(label_opose_bytes)
        label_opose = Image.open(buffer).convert('RGB')

        if self.training and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            opose = opose.transpose(Image.FLIP_LEFT_RIGHT)
            label_opose = label_opose.transpose(Image.FLIP_LEFT_RIGHT)

        opose = self.transforms(opose)
        img = self.transforms(img)
        label_opose = self.transforms(label_opose)

        input_dict = {'label': opose, 'image': img, 'style': label_opose, 'name': imgname}

        return input_dict

    def __len__(self):
        return self.length

    def get_resolution(self):
        return self.resolution

    def set_resolution(self, resolution):
        self.resolution = resolution
