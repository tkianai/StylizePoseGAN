"""Training Pairs: (image, openpose)
"""

import os
import argparse
import lmdb
from PIL import Image
from io import BytesIO
from functools import partial
import multiprocessing as mp
from tqdm import tqdm
import numpy as np
import pickle
import cv2
import random


_OPENPOSE = {
    "pose": [
        # ((0, 1, 8), (255, 0, 0)),
        ((1, 2, 3, 4), (0, 255, 0)),
        ((1, 5, 6, 7), (0, 0, 255)),
        # ((8, 9, 10), (255, 255, 0)),
        # ((8, 12, 13), (255, 0, 255)),
    ],
    "face": [
        (list(range(0, 17)), (255, 255, 255)),
        (list(range(17, 22)), (255, 255, 255)),
        (list(range(22, 27)), (255, 255, 255)),
        (list(range(27, 31)), (255, 255, 255)),
        (list(range(31, 36)), (255, 255, 255)),
        (list(range(36, 42)), (255, 255, 255)),
        (list(range(42, 48)), (255, 255, 255)),
        (list(range(48, 60)), (255, 255, 255)),
        (list(range(60, 68)), (255, 255, 255)),
    ],
    "hand": [
        ((0, 17, 18, 19, 20), (160, 82, 45)),
        ((0, 13, 14, 15, 16), (50, 205, 50)),
        ((0, 9, 10, 11, 12), (65, 105, 225)),
        ((0, 5, 6, 7, 8), (148, 0, 211)),
        ((0, 1, 2, 3, 4), (255, 192, 203))
    ]
}


def openpose_keypoints_to_image(file, size=1024):

    with open(file, 'rb') as r_obj:
        pose = pickle.load(r_obj)

    if not isinstance(size, (list, tuple)):
        size = (size, size)
    size = (size[0], size[1], 3)
    img = np.zeros(size, dtype=np.uint8)

    for key, value in _OPENPOSE.items():

        linewidth = 10 if key == 'pose' else 4
        threshold = 0.5 if key == 'hand' else 0.3
        pose_item = pose[key]
        if not isinstance(pose_item, list):
            pose_item = [pose_item]

        for itm_data in pose_item:
            for itm_con in value:
                points, color = itm_con

                for i in range(len(points) - 1):
                    if itm_data[0, points[i], 2] > threshold and itm_data[0, points[i+1], 2] > threshold:
                        cv2.line(img, (int(itm_data[0, points[i], 0]), int(itm_data[0, points[i], 1])),
                                 (int(itm_data[0, points[i+1], 0]), int(itm_data[0, points[i+1], 1])), color, linewidth)

    img = Image.fromarray(img)
    return img


def resize_to_bytes(img, size, quality):

    crop_img = img.resize((size, size), Image.LANCZOS)
    buffer = BytesIO()
    crop_img.save(buffer, format='jpeg', quality=quality)
    value = buffer.getvalue()

    return value


def resize_into_resolutions(img, sizes, quality=100):
    imgs = []
    for size in sizes:
        imgs.append(resize_to_bytes(img, size, quality))

    return imgs


def resize_job(img_file, sizes):
    i, file, opose = img_file
    img = Image.open(file)
    img = img.convert('RGB')
    imgs = resize_into_resolutions(img, sizes=sizes)

    # process for openpose results
    opose = openpose_keypoints_to_image(opose)
    buffer = BytesIO()
    opose.save(buffer, format='jpeg', quality=100)
    opose = buffer.getvalue()

    return i, imgs, opose


def write_to_lmdb(txn, dset, worker, sizes=(8, 16, 32, 64, 128, 256, 512, 1024)):

    files = [(i, img, opose) for i, (img, opose) in enumerate(dset)]
    total = 0

    resize_fn = partial(resize_job, sizes=sizes)

    with mp.Pool(worker) as pool:
        for i, imgs, opose in tqdm(pool.imap_unordered(resize_fn, files)):
            for size, img in zip(sizes, imgs):
                key = 'Image-{}-{:0>7d}'.format(size, i).encode('utf-8')
                txn.put(key, img)

            key = 'Openpose-{:0>7d}'.format(i).encode('utf-8')
            txn.put(key, opose)

            # process one image
            total += 1

        # processing done
        txn.put('Length'.encode('utf-8'), str(total).encode('utf-8'))


def is_image(file, image_set=('png', 'jpg', 'jpeg')):
    flag = False
    if file.split('.')[-1].lower() in image_set:
        flag = True
    return flag


def main(args):

    dset = []
    for img in os.listdir(args.imgs):
        if is_image(img):
            imgpath = os.path.join(args.imgs, img)

            opose = '.'.join(img.split('.')[:-1]) + '.pkl'
            oposepath = os.path.join(args.openpose, opose)

            if os.path.isfile(imgpath) and os.path.isfile(oposepath):
                dset.append((imgpath, oposepath))

    random.shuffle(dset)
    train_num = int(len(dset) * 0.92)
    dset_train = dset[:train_num]
    dset_test = dset[train_num:]

    # write to lmdb
    with lmdb.open(args.out + '_train.lmdb', map_size=1024**4) as env:
        with env.begin(write=True) as txn:
            write_to_lmdb(txn, dset_train, args.worker)

    # write to lmdb
    with lmdb.open(args.out + '_test.lmdb', map_size=1024**4) as env:
        with env.begin(write=True) as txn:
            write_to_lmdb(txn, dset_test, args.worker)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create multi-resolution pose dataset")
    parser.add_argument(
        '--out', type=str, default='datasets/pose', help='Output dataset path')
    parser.add_argument('--worker', type=int, default=10,
                        help='Number of multi-process cores')
    parser.add_argument('--imgs', type=str,
                        default='datasets/yangdan/image', help='Image directory')
    parser.add_argument('--openpose', type=str,
                        default='datasets/yangdan/openpose', help='Openpose results')

    args = parser.parse_args()
    if not os.path.exists(os.path.dirname(args.out)):
        os.makedirs(os.path.dirname(args.out))
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
