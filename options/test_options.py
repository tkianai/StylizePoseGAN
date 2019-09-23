
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--test_size', type=int, default=1024, help='test image size')
        self.parser.add_argument(
            '--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--which_iter', type=str, default='latest',
                                 help='which iteration to load? set to latest to use latest cached model')
        
        self.isTrain = False
