from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # for displays
        self.parser.add_argument('--no_html', action='store_true',
                                 help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        # for training
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--load_pretrain', type=str, default='',
                                 help='load the pretrained model from the specified location')
        self.parser.add_argument('--which_iter', type=str, default='latest',
                                 help='which iteration to load? set to latest to use latest cached model')

        # for discriminators
        self.parser.add_argument('--no_ganFeat_loss', action='store_true',
                                 help='if specified, do *not* use discriminator feature matching loss')
        self.parser.add_argument('--no_vgg_loss', action='store_true',
                                 help='if specified, do *not* use VGG feature matching loss')
        self.parser.add_argument('--no_lsgan', action='store_true',
                                 help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--no_gp', action='store_true',
                                 help='do *not* use gradient penalty')

        self.parser.add_argument(
            '--lambda_feat', type=float, default=10.0, help='weight for feature matching loss')
        self.parser.add_argument(
            '--lambda_gp', type=float, default=5.0, help='weight for gradient penalty')
        self.parser.add_argument(
            '--lr', type=float, default=0.0002, help='initial learning rate for adam')

        self.parser.add_argument(
            '--min_size', type=int, default=8, help='minimum training size')
        self.parser.add_argument(
            '--max_size', type=int, default=1024, help='maximum training size')
        self.parser.add_argument(
            '--start_size', type=int, default=8, help='start training size')
        self.parser.add_argument(
            '--size_iter', type=int, default=1, help='iteration counter for each step')
        self.parser.add_argument(
            '--start_epoch', type=int, default=1, help='record epoch for whole training')
        self.parser.add_argument(
            '--start_iter', type=int, default=1, help='the whole iteration counter')
        self.parser.add_argument(
            '--max_iter', type=int, default=25000, help='max iteration for training')
        self.parser.add_argument(
            '--iter_each_step', type=int, default=3000, help='iterations for each step')
        
        self.parser.add_argument(
            '--print_freq', type=int, default=10, help='print error each # iterations')
        self.parser.add_argument(
            '--display_freq', type=int, default=100, help='display sampled images each # iterations')
        self.parser.add_argument(
            '--save_freq', type=int, default=1000, help='save checkpoint each # iterations')

        self.isTrain = True
