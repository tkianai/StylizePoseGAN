
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
            self.log_dir = os.path.join(opt.checkpoints_dir, opt.name, 'runs')
            self.writer = SummaryWriter(self.log_dir)

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print("create web directory {}...".format(self.web_dir))
            misc.mkdirs([self.web_dir, self.img_dir])

        self.log_name = os.path.join(
            opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write(
                '======== Training Loss ({}) ========\n'.format(now))

    def display_current_results(self, visuals, size, epoch, iteration):
        if self.tboard:
            img_summaries = []
            for label, image_numpy in visuals.items():
                self.writer.add_image(
                    "size-{}-epoch-{}-iter-{}-{}.jpg".format(size, epoch, iteration, label), image_numpy, dataformats='HWC')

        if self.use_html:
            for label, image_numpy in visuals.items():
                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        img_path = os.path.join(
                            self.img_dir, "size-{}-epoch-{}-iter-{}-{}_{}.jpg".format(size, epoch, iteration, label, i))
                        misc.save_image(image_numpy[i], img_path)

                else:
                    img_path = os.path.join(
                        self.img_dir, "size-{}-epoch-{}-iter-{}-{}.jpg".format(size, epoch, iteration, label))
                    misc.save_image(image_numpy, img_path)

            # update webpage
            webpage = html.HTML(self.web_dir, "Experiment name = {}".format(self.name), refresh=30)
            images = sorted(os.listdir(self.img_dir), reverse=True)
            webpage.add_header("Training process visulization")
            ims = []
            txts = []
            links = []
            for image in images:
                ims.append(image)
                links.append(image)
                label = image.split('.')[0].split('_')[0].split('-')[-1]
                label += ": " + image.split('.')[0].replace('_', '@')
                txts.append(label)
            
            if len(ims) < 10:
                    webpage.add_images(ims, txts, links, width=self.win_size)
            else:
                num = int(round(len(ims) / 2.0))
                webpage.add_images(
                    ims[:num], txts[:num], links[:num], width=self.win_size)
                webpage.add_images(
                    ims[num:], txts[num:], links[num:], width=self.win_size)

            webpage.save()

    def plot_current_errors(self, errors, iteration):
        if self.tboard:
            for tag, value in errors.items():
                self.writer.add_scalar(tag, value, iteration)

    def print_current_errors(self, size, epoch, iteration, errors, t):
        msg = "(size: {}, epoch: {}, iters: {}, time: {:.3f})".format(size, epoch, iteration, t)
        for k, v in errors.items():
            if v != 0:
                msg += "{}: {:.3f} ".format(k, v)

        print(msg)
        with open(self.log_name, "a") as log_file:
            log_file.write("{}\n".format(msg))
