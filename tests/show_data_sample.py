

import os
import argparse
import lmdb
import random
from io import BytesIO
from PIL import Image


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--data')
    parser.add_argument('--save')

    args = parser.parse_args()
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    return args


def main(args):
    
    sizes = (8, 16, 32, 64, 128, 256, 512, 1024)

    env = lmdb.open(args.data, readonly=True, lock=False, meminit=False)
    if not env:
        raise IOError("Cannot open lmdb dataset", args.data)

    with env.begin(write=False) as txn:
        length = int(txn.get("Length".encode('utf-8')).decode('utf-8'))
        print("Dataset samples: {}".format(length))
        
        idx = random.randint(0, length)
        for size in sizes:
            key = 'Image-{}-{:0>7d}'.format(size, idx).encode('utf-8')
            img_bytes = txn.get(key)
            buffer = BytesIO(img_bytes)
            img = Image.open(buffer).convert('RGB')
            img.save(os.path.join(args.save, 'Image-{}-{:0>7d}.jpg'.format(size, idx)))

        key = 'Openpose-{:0>7d}'.format(idx).encode('utf-8')
        pose_bytes = txn.get(key)
        buffer = BytesIO(pose_bytes)
        pose = Image.open(pose_bytes).convert('RGB')
        pose.save(os.path.join(args.save, 'Openpose-{:0>7d}.jpg'.format(idx)))
        
        

if __name__ == "__main__":
    args = parse_args()
    main(args)
