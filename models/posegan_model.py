
import numpy as np
import torch
import os
from .base_model import BaseModel
from . import posegan_util


class PoseGANModel(BaseModel):

    def name(self):
        return "PoseGANModel"


    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, use_gp):
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True, use_gp)

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake, gp):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake, gp), flags) if f]

        return loss_filter


    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if not opt.isTrain:
            torch.backends.cudnn.benchmark = True

        # define networks
        self.netG = posegan_util.define_G(code_dim=opt.code_dim, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.netD = posegan_util.define_D(gpu_ids=self.gpu_ids)
        print("---------------Network created-------------")

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)
        
        # set loss functions and optimizers
        if self.isTrain:
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, not opt.no_gp)
            self.criterionGAN = posegan_util.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()

            if not opt.no_vgg_loss:
                self.criterionVGG = posegan_util.VGGLoss(self.gpu_ids)

            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake', 'GP')

            # initialize optimizer
            self.optimizer_G = torch.optim.Adam(self.netG.generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_G.add_param_group({
                'params': self.netG.style.parameters(),
                'lr': opt.lr * 0.1,
            })
            self.optimizer_D = torch.optim.Adam(self.netD.discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

    def encode_input(self, label_map, real_image=None):

        input_label = label_map.cuda()
        if real_image is not None:
            real_image = real_image.cuda()

        return input_label, real_image

    def discriminate(self, input_label, test_image, step, alpha):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        return self.netD(input_concat, step=step, alpha=alpha)


    def forward(self, label, image, step=0, alpha=-1, infer=False):

        input_label, real_image = self.encode_input(label, image)

        # Fake Generation
        fake_image = self.netG(input_label, step=step, alpha=alpha)

        # Fake detection and Loss
        pred_fake = self.discriminate(input_label, fake_image, step, alpha)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real Detection and Loss
        pred_real = self.discriminate(input_label, real_image, step, alpha)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake passability loss)
        pred_fake_for_g = self.netD(torch.cat((input_label, fake_image), dim=1))
        loss_G_GAN = self.criterionGAN(pred_fake_for_g, True)

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 1.0 / len(pred_fake)
            for i in range(len(pred_fake) - 1):
                loss_G_GAN_Feat += feat_weights * self.opt.lambda_feat * self.criterionFeat(pred_fake[i], pred_real[i].detach())

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat

        # gradient penalty
        loss_gp = 0
        if not self.opt.no_gp:
            real_image.requires_grad = True
            pred_real_gp = self.discriminate(input_label, real_image, step, alpha)
            grad_real = torch.autograd.grad(
                outputs=pred_real_gp.sum(), 
                inputs=real_image, 
                create_graph=True
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            loss_gp = self.opt.lambda_gp * grad_penalty
            real_image.requires_grad = False


        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake, loss_gp), None if not infer else fake_image]


    def inference(self, label, step=0, alpha=-1):

        input_label, real_image = self.encode_input(label, image)

        # fake generation
        with torch.no_grad():
            fake_image = self.netG(input_label, step=step, alpha=alpha)

        return fake_image

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)


    def update_learning_rate(self, lr):
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr