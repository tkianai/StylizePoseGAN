
import numpy as np
import torch
import os
from .base_model import BaseModel
from . import posegan_util


class PoseGANModel(BaseModel):

    def name(self):
        return "PoseGANModel"


    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if not opt.isTrain:
            torch.backends.cudnn.benchmark = True

        # define networks
        self.netG = posegan_util.define_G(code_dim=opt.code_dim, gpu_ids=self.gpu_ids)

        if self.isTrain:
            self.netD = posegan_util.define_D(gpu_ids=self.gpu_ids)
        print("---------------Network created-------------")

        # set loss functions and optimizers
        if self.isTrain:
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = posegan_util.GANLoss(
                use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()

            if not opt.no_vgg_loss:
                self.criterionVGG = posegan_util.VGGLoss(self.gpu_ids)

            # initialize optimizer
            self.optimizer_G = torch.optim.Adam(
                self.netG.generator.parameters(), lr=opt.lr, betas=(0.5, 0.999))
            self.optimizer_G.add_param_group({
                'params': self.netG.style.parameters(),
                'lr': opt.lr,
            })
            self.optimizer_D = torch.optim.Adam(
                self.netD.discriminator.parameters(), lr=opt.lr, betas=(0.5, 0.999))

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.netG, 'G', opt.which_iter, pretrained_path)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_iter, pretrained_path)

    def encode_input(self, style, label_map=None, real_image=None):

        input_style = style.cuda()
        if label_map is not None:
            input_label = label_map.cuda()
        if real_image is not None:
            real_image = real_image.cuda()

        return input_style, input_label, real_image

    def discriminate(self, input_label, test_image, step, alpha):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        return self.netD(input_concat, step=step, alpha=alpha)


    def forward(self, style, label, image, step=0, alpha=-1, infer=False):

        losses = {}
        input_style, input_label, real_image = self.encode_input(style, label, image)

        # Fake Generation
        fake_image = self.netG(input_style, step=step, alpha=alpha)

        # Fake detection and Loss
        pred_fake = self.discriminate(input_label, fake_image, step, alpha)
        loss_D_fake = self.criterionGAN(pred_fake, False)
        losses['D_fake'] = loss_D_fake

        # Real Detection and Loss
        pred_real = self.discriminate(input_label, real_image, step, alpha)
        loss_D_real = self.criterionGAN(pred_real, True)
        losses['D_real'] = loss_D_real

        # GAN loss (Fake passability loss)
        pred_fake_for_g = self.netD(torch.cat((input_label, fake_image), dim=1), step, alpha)
        loss_G_GAN = self.criterionGAN(pred_fake_for_g, True)
        losses['G_GAN'] = loss_G_GAN

        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 1.0 / len(pred_fake)
            for i in range(len(pred_fake) - 1):
                loss_G_GAN_Feat += feat_weights * self.opt.lambda_feat * self.criterionFeat(pred_fake[i], pred_real[i].detach())
            losses['G_GAN_Feat'] = loss_G_GAN_Feat

        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss and step > 4:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
            losses['G_VGG'] = loss_G_VGG

        # gradient penalty
        loss_gp = 0
        if not self.opt.no_gp:
            real_image.requires_grad = True
            pred_real_gp = self.discriminate(input_label, real_image, step, alpha)
            grad_real = torch.autograd.grad(
                outputs=pred_real_gp[-1].sum(), 
                inputs=torch.cat((input_label, real_image), dim=1),
                create_graph=True,
                allow_unused=True,
                retain_graph=True,
            )[0]
            grad_penalty = (
                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
            ).mean()
            loss_gp = self.opt.lambda_gp * grad_penalty
            real_image.requires_grad = False
            losses['GP'] = loss_gp

        return [losses, None if not infer else fake_image]


    def inference(self, style, step=0):

        input_style, input_label, real_image = self.encode_input(style, None, None)

        # fake generation
        with torch.no_grad():
            fake_image = self.netG(input_style, step=step, alpha=1)

        return fake_image

    def save(self, which_size, which_iter):
        self.save_network(self.netG, 'G', which_size, which_iter, self.gpu_ids)
        self.save_network(self.netD, 'D', which_size, which_iter, self.gpu_ids)


    def update_learning_rate(self, lr):
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
