
import os
import math
from options.test_options import TestOptions
from utils import misc
from data.build import build_dataloader
from models import build_model
from collections import OrderedDict


def test(model, data_loader, save_dir):
    dataset = data_loader.load_data()
    step = int(math.log2(dataset.dataset.get_resolution())) - 2

    for i, data in enumerate(dataset):
        generated = model.inference(data['style'], step=step)
        visuals = OrderedDict([('input_label', misc.tensor2im(data['style'][0])), ('synthesized_image', misc.tensor2im(generated.data[0]))])
        name = data['name'][0].split('/')[-1].split('.')[0]
        misc.save_images(visuals, name, save_dir)


def main(opt):
    data_loader = build_dataloader(opt.dataroot, opt.label_size, training=opt.isTrain, resolution=opt.test_size, batch_size=1)
    model = build_model(opt)
    save_dir = os.path.join(opt.results_dir, opt.name, 'test_{}_{}'.format(opt.test_size, opt.which_iter))
    misc.mkdir(save_dir)
    test(model, data_loader, save_dir)


if __name__ == "__main__":
    opt = TestOptions().parse(save=False)
    main(opt)
