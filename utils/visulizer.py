
import os
import numpy as np
import ntpath
import time
import scipy.misc
from . import misc
from . import html
from torch.utils.tensorboard import SummaryWriter


class Visualizer():
    def __init__(self, opt):

        self.tboard = opt.tboard
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name

        if self.tboard:
            self.log_idr = os.path.join(opt.checkpoints_dir, opt.name, 'runs')
            self.writer = SummaryWriter(self.log_idr)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print("create web directory {}...".format(self.web_dir))
            misc.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(
            opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_idr, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(
                '======== Training Loss ({}) ========\n'.format(now))

    def display_current_results(self, visuals, epoch, step):
        if self.tboard:
            img_summaries = []
            for label, image_numpy in visuals.items():
                self.writer.add_image(
                    "Epoch{}step-{}-{}".format(epoch, step, label), image_numpy, dataformats='HWC')

        if self.use_html:
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(
                            self.img_dir, "epoch{}_{}_{}.jpg".format(epoch, label, i))
                        misc.save_image(image_numpy[i], img_path)

                else:
                    img_path = os.path.join(
                        self.img_dir, "epoch{}_{}.jpg".format(epoch, label))
                    misc.save_image(image_numpy, img_path)

            # update webpage
            webpage = html.HTML(
                self.web_dir, "Experiment name = {}".format(name), refresh=30)
            for n in range(epoch, 0, -1):
                webpage.add_header("epoch [{}]".format(n))
                ims = []
                txts = []
                links = []

                for label, image_numpy in visuals.items():
                    if isinstance(image_numpy, list):
                        for i in range(len(image_numpy)):
                            img_path = "epoch{}_{}_{}.jpg".format(n, label, i)
                            ims.append(img_path)
                            txts.append(label + str(i))
                            links.append(img_path)
                    else:
                        img_path = "epoch{}_{}.jpg".format(n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)

                if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
                else:
                    num = int(round(len(ims) / 2.0))
                    webpage.add_images(
                        ims[:num], txts[:num], links[:num], width=self.win_size)
                    webpage.add_images(
                        ims[num:], txts[num:], links[num:], width=self.win_size)

            webpage.save()

    def plot_current_errors(self, errors, step):
        if self.tboard:
            for tag, value in errors.items():
                self.writer.add_scalar(tag, value)

    def print_current_errors(self, epoch, i, errors, t):
        msg = "(epoch: {}, iters: {}, time: {:.3f})".format(epoch, i, t)
        for k, v in errors.items():
            if v != 0:
                msg += "{}: {:.3f}".format(k, v)

        print(msg)
        with open(self.log_name, "a") as log_file:
            log_file.write("{}\n".format(msg))

    # save image to the disk
    def save_images(self, webpage, visuals, image_path):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.jpg' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size)
