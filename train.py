
import os
import time
import numpy as np
import torch
from collections import OrderedDict

from options.train_options import TrainOptions
from data.build import build_dataloader
from models import build_model
from utils import misc
from utils.visulizer import Visualizer


def train(model, optimizer_G, optimizer_D, data_loader, train_solver, visualizer, start_step, start_epoch, iter_path):
    max_step = 8
    min_step = 1
    print_freq = 100
    display_freq = 100
    save_freq = 1000
    optimize_step = 1
    
    for step in range(start_step, max_step + 1):

        # config for each step
        resolution = 4 * 2 ** step
        config = train_solver.get(resolution)
        batch_size = config['batch_size']
        epochs = config['epochs']
        lr = config['lr']

        # update learning rate
        model.module.update_learning_rate(lr)

        # setup dataset
        data_loader.update(resolution, batch_size)
        dataset = data_loader.load_data()
        dataset_size = len(dataset)

        # train
        for epoch in range(start_epoch, epochs):
            if (epoch > epochs // 2) or step == min_step:
                alpha = 1
            else:
                # NOTE another choice, update in iteration level
                alpha = min(1, 1.0 / (epochs // 2) * (epoch + 1))
            epoch_start_time = time.time()
            for i, data in enumerate(dataset):
                optimize_step += i
                if optimize_step % print_freq == 0:
                    iter_start_time = time.time()
                
                # whether to collect output images
                save_fake = optimize_step % display_freq == 0

                ############## Forward Pass ######################
                losses, generated = model(data['label'], data['image'], step=step, alpha=alpha, infer=save_fake)

                # sum per device losses
                losses = [torch.mean(x) if not isinstance(x, int) else x for x in losses]
                loss_dict = dict(zip(model.module.loss_names, losses))

                # calculate final loss scalar
                loss_D = (loss_dict['D_fake'] + loss_dict['D_real']) * 0.5
                loss_G = loss_dict['G_GAN'] + loss_dict.get('G_GAN_Feat', 0) + loss_dict.get('G_VGG', 0)
                loss_GP = loss_dict.get('loss_gp', 0)

                ############### Backward Pass ####################
                # update generator weights
                optimizer_G.zero_grad()
                loss_G.backward()
                optimizer_G.step()

                # update discriminator weights
                optimizer_D.zero_grad()
                loss_D.backward()
                loss_GP.backward()
                optimizer_D.step()

                ############## Display results and errors ##########
                ### print out errors
                if optimize_step % print_freq == 0:
                    errors = {k: v.data.item() if not isinstance(
                        v, int) else v for k, v in loss_dict.items()}
                    t = (time.time() - iter_start_time) / print_freq
                    visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                    visualizer.plot_current_errors(errors, optimize_step)


                ### display output images
                if save_fake:
                    visuals = OrderedDict([('input_label', misc.tensor2im(data['label'][0])),
                                        ('synthesized_image', misc.tensor2im(generated.data[0])),
                                        ('real_image', misc.tensor2im(data['image'][0]))])
                    visualizer.display_current_results(visuals, epoch, optimize_step)

                ### save latest model
                if optimize_step % save_freq == 0:
                    print('saving the latest model (epoch %d, optimize_step %d)' %
                        (epoch, optimize_step))
                    model.module.save('latest')
                    model.module.save("{}_{}".format(step, epoch))
                    np.savetxt(iter_path, (step, epoch), delimiter=',', fmt='%d')

            # end of epoch
            print('End of epoch %d / %d \t Time Taken: %d sec' %
                (epoch, epochs, time.time() - epoch_start_time))


def main(opt):

    iter_path = os.path.join(opt.checkpoints_dir, opt.name, 'iter.txt')
    if opt.continue_train:
        try:
            start_step, start_epoch = np.loadtxt(iter_path, delimiter=',', dtype=int)
        except:
            start_step, start_epoch = 1, 1
        print('Resuming from epoch {} at iteration {}'.format(start_epoch, epoch_iter))
    else:
        start_step, start_epoch = 1, 1

    data_loader = build_dataloader(opt.dataroot, training=opt.isTrain, resolution=8, batch_size=256)

    model = build_model(opt)
    visualizer = Visualizer(opt)
    optimizer_G, optimizer_D = model.module.optimizer_G, model.module.optimizer_D

    train_solver = {
        8: {'batch_size': 256, 'epochs': 128, 'lr': 0.0002},
        16: {'batch_size': 256, 'epochs': 128, 'lr': 0.0002},
        32: {'batch_size': 256, 'epochs': 128, 'lr': 0.0002},
        64: {'batch_size': 128, 'epochs': 128, 'lr': 0.0002},
        128: {'batch_size': 128, 'epochs': 64, 'lr': 0.0002},
        256: {'batch_size': 64, 'epochs': 32, 'lr': 0.0002},
        512: {'batch_size': 32, 'epochs': 32, 'lr': 0.0002},
        1024: {'batch_size': 16, 'epochs': 32, 'lr': 0.0002},
    }

    train(model, optimizer_G, optimizer_D, data_loader, train_solver, visualizer, start_step, start_epoch, iter_path)


if __name__ == "__main__":
    opt = TrainOptions().parse()
    main(opt)
