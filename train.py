
import os
import math
import time
import datetime
import numpy as np
import torch
from collections import OrderedDict

from options.train_options import TrainOptions
from data.build import build_dataloader
from models import build_model
from utils import misc
from utils.visulizer import Visualizer


def config_step(step, size_solver, model, data_loader):

    # config for each step
    resolution = 4 * 2 ** step
    config = size_solver.get(resolution)
    batch_size = config['batch_size']
    lr = config['lr']

    # update learning rate
    model.module.update_learning_rate(lr)

    # setup dataset
    data_loader.update(resolution, batch_size)
    dataset = data_loader.load_data()
    
    return dataset, resolution


def train(opt, model, optimizer_G, optimizer_D, data_loader, size_solver, visualizer, iter_path):

    step = int(math.log2(opt.start_size)) - 2
    max_step = int(math.log2(opt.max_size)) - 2
    iteration = opt.start_iter
    epoch = opt.start_epoch
    size_iter = opt.size_iter
    APPROACHED = True if step > max_step else False
    iter_start_time = time.time()
    start_training_time = time.time()
    dataset, resolution = config_step(step, size_solver, model, data_loader)

    while True:

        for data in dataset:
            
            if (size_iter > opt.iter_each_step) or (iteration > opt.max_iter):
                break

            if (size_iter > opt.iter_each_step // 2) or (resolution == opt.min_size) or APPROACHED:
                alpha = 1
            else:
                alpha = min(1, 1.0 / (opt.iter_each_step // 2) * (size_iter + 1))

            
            # whether to collect output images
            save_fake = iteration % opt.display_freq == 0

            losses, generated = model(
                data['style'], data['label'], data['image'], step=step, alpha=alpha, infer=save_fake)

            # sum per device losses
            loss_dict = {k: torch.mean(v) for k, v in losses.items()}

            # calculate final loss scalar
            loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
            loss_G = loss_dict['G_GAN'] + \
                loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)
            loss_GP = loss_dict.get('GP', 0)

            # update generator weights
            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # update discriminator weights
            optimizer_D.zero_grad()
            loss_D.backward()
            if not opt.no_gp:
                loss_GP.backward()
            optimizer_D.step()

            # print out errors
            if iteration % opt.print_freq == 0:
                errors = {k: v.data.item() if not isinstance(
                    v, int) else v for k, v in loss_dict.items()}
                t = (time.time() - iter_start_time) / opt.print_freq
                visualizer.print_current_errors(resolution, epoch, iteration, errors, t)
                visualizer.plot_current_errors(errors, iteration)
                iter_start_time = time.time()

            # display output images
            if save_fake:
                visuals = OrderedDict([('input_label', misc.tensor2im(data['label'][0])),
                                        ('synthesized_image', misc.tensor2im(generated.data[0])),
                    ('real_image', misc.tensor2im(data['image'][0]))])
                visualizer.display_current_results(
                    visuals, resolution, epoch, iteration)

            # save latest model
            if iteration % opt.save_freq == 0:
                print('saving the latest model (size {}, iteration {})'.format(resolution, iteration))
                model.module.save('0', 'latest')
                model.module.save(str(resolution), str(iteration))
                np.savetxt(iter_path, (resolution, size_iter, epoch, iteration), delimiter=',', fmt='%d')

            size_iter += 1
            iteration += 1
        
        epoch += 1

        if size_iter > opt.iter_each_step:
            # update config
            step += 1
            if step > max_step:
                step = max_step
                APPROACHED = True

            size_iter = 0
            dataset, resolution = config_step(step, size_solver, model, data_loader)
        
        if iteration > opt.max_iter:
            total_training_time = time.time() - start_training_time
            total_time_str = str(datetime.timedelta(seconds=total_training_time))
            print("Total training time: {} ({:.4f} s / it)".format(
                total_time_str, total_training_time / (opt.max_iter - opt.start_iter)
            ))
            model.module.save('Final-' + str(resolution), str(iteration))
            break


def main(opt):

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_size, size_iter, start_epoch, start_iter = np.loadtxt(iter_path, delimiter=',', dtype=int)
            print('Resuming from size {} at iteration {}'.format(
                start_size, start_iter))
            opt.start_size = start_size
            opt.size_iter = size_iter
            opt.start_epoch = start_epoch
            opt.start_iter = start_iter
        except:
            print('Start training from scratch: size {} at iteration {}'.format(
                opt.start_size, opt.start_iter))

    data_loader = build_dataloader(opt.dataroot, opt.label_size, training=opt.isTrain, resolution=8, batch_size=8)
    model = build_model(opt)
    visualizer = Visualizer(opt)
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

    size_solver = {
        8: {'batch_size': 40, 'lr': 0.001},
        16: {'batch_size': 40, 'lr': 0.001},
        32: {'batch_size': 32, 'lr': 0.001},
        64: {'batch_size': 32, 'lr': 0.002},
        128: {'batch_size': 24, 'lr': 0.002},
        256: {'batch_size': 16, 'lr': 0.002},
        512: {'batch_size': 8, 'lr': 0.003},
        1024: {'batch_size': 8, 'lr': 0.003},
    }

    train(opt, model, optimizer_G, optimizer_D, data_loader, size_solver, visualizer, iter_path)


if __name__ == "__main__":
    opt = TrainOptions().parse()
    main(opt)
