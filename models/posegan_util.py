
from torchvision import models
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.autograd import Function
from math import sqrt
import random


class EqualLR:
    def __init__(self, name):
        self.name = name

    def compute_weight(self, module):
        weight = getattr(module, self.name + '_orig')
        fan_in = weight.data.size(1) * weight.data[0][0].numel()

        return weight * sqrt(2 / fan_in)

    @staticmethod
    def apply(module, name):
        fn = EqualLR(name)

        weight = getattr(module, name)
        del module._parameters[name]
        module.register_parameter(name + '_orig', nn.Parameter(weight.data))
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, input):
        weight = self.compute_weight(module)
        setattr(module, self.name, weight)


def equal_lr(module, name='weight'):
    EqualLR.apply(module, name)

    return module


class FusedUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv_transpose2d(
            input, weight, self.bias, stride=2, padding=self.pad)

        return out


class FusedDownsample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, padding=0):
        super().__init__()

        weight = torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        bias = torch.zeros(out_channel)

        fan_in = in_channel * kernel_size * kernel_size
        self.multiplier = sqrt(2 / fan_in)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        self.pad = padding

    def forward(self, input):
        weight = F.pad(self.weight * self.multiplier, [1, 1, 1, 1])
        weight = (
            weight[:, :, 1:, 1:]
            + weight[:, :, :-1, 1:]
            + weight[:, :, 1:, :-1]
            + weight[:, :, :-1, :-1]
        ) / 4

        out = F.conv2d(input, weight, self.bias, stride=2, padding=self.pad)

        return out


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input / torch.sqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


class BlurFunctionBackward(Function):
    @staticmethod
    def forward(ctx, grad_output, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        grad_input = F.conv2d(
            grad_output, kernel_flip, padding=1, groups=grad_output.shape[1]
        )

        return grad_input

    @staticmethod
    def backward(ctx, gradgrad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = F.conv2d(
            gradgrad_output, kernel, padding=1, groups=gradgrad_output.shape[1]
        )

        return grad_input, None, None


class BlurFunction(Function):
    @staticmethod
    def forward(ctx, input, kernel, kernel_flip):
        ctx.save_for_backward(kernel, kernel_flip)

        output = F.conv2d(input, kernel, padding=1, groups=input.shape[1])

        return output

    @staticmethod
    def backward(ctx, grad_output):
        kernel, kernel_flip = ctx.saved_tensors

        grad_input = BlurFunctionBackward.apply(
            grad_output, kernel, kernel_flip)

        return grad_input, None, None


blur = BlurFunction.apply


class Blur(nn.Module):
    def __init__(self, channel):
        super().__init__()

        weight = torch.tensor(
            [[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float32)
        weight = weight.view(1, 1, 3, 3)
        weight = weight / weight.sum()
        weight_flip = torch.flip(weight, [2, 3])

        self.register_buffer('weight', weight.repeat(channel, 1, 1, 1))
        self.register_buffer(
            'weight_flip', weight_flip.repeat(channel, 1, 1, 1))

    def forward(self, input):
        return blur(input, self.weight, self.weight_flip)
        # return F.conv2d(input, self.weight, padding=1, groups=input.shape[1])


class EqualConv2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        conv = nn.Conv2d(*args, **kwargs)
        conv.weight.data.normal_()
        conv.bias.data.zero_()
        self.conv = equal_lr(conv)

    def forward(self, input):
        return self.conv(input)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()

        linear = nn.Linear(in_dim, out_dim)
        linear.weight.data.normal_()
        linear.bias.data.zero_()

        self.linear = equal_lr(linear)

    def forward(self, input):
        return self.linear(input)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        padding,
        kernel_size2=None,
        padding2=None,
        downsample=False,
        fused=False,
    ):
        super().__init__()

        pad1 = padding
        pad2 = padding
        if padding2 is not None:
            pad2 = padding2

        kernel1 = kernel_size
        kernel2 = kernel_size
        if kernel_size2 is not None:
            kernel2 = kernel_size2

        self.conv1 = nn.Sequential(
            EqualConv2d(in_channel, out_channel, kernel1, padding=pad1),
            nn.LeakyReLU(0.2),
        )

        if downsample:
            if fused:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    FusedDownsample(out_channel, out_channel,
                                    kernel2, padding=pad2),
                    nn.LeakyReLU(0.2),
                )

            else:
                self.conv2 = nn.Sequential(
                    Blur(out_channel),
                    EqualConv2d(out_channel, out_channel,
                                kernel2, padding=pad2),
                    nn.AvgPool2d(2),
                    nn.LeakyReLU(0.2),
                )

        else:
            self.conv2 = nn.Sequential(
                EqualConv2d(out_channel, out_channel, kernel2, padding=pad2),
                nn.LeakyReLU(0.2),
            )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        return out


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self, in_channel, style_dim):
        super().__init__()

        self.norm = nn.InstanceNorm2d(in_channel)
        self.style = EqualLinear(style_dim, in_channel * 2)

        self.style.linear.bias.data[:in_channel] = 1
        self.style.linear.bias.data[in_channel:] = 0

    def forward(self, input, style):
        style = self.style(style).unsqueeze(2).unsqueeze(3)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return out


class NoiseInjection(nn.Module):
    def __init__(self, channel):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1, channel, 1, 1))

    def forward(self, image, noise):
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size=3,
        padding=1,
        style_dim=512,
        initial=False,
        upsample=False,
        fused=False,
    ):
        super().__init__()

        if initial:
            self.conv1 = ConstantInput(in_channel)
        else:
            if upsample:
                if fused:
                    self.conv1 = nn.Sequential(
                        FusedUpsample(
                            in_channel,
                            out_channel,
                            kernel_size,
                            padding=padding,
                        ),
                        Blur(out_channel),
                    )
                else:
                    self.conv1 = nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        EqualConv2d(
                            in_channel,
                            out_channel,
                            kernel_size,
                            padding=padding
                        ),
                        Blur(out_channel),
                    )
            else:
                self.conv1 = EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=padding,
                )
                
        self.noise1 = equal_lr(NoiseInjection(out_channel))
        self.adain1 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu1 = nn.LeakyReLU(0.2)

        self.conv2 = EqualConv2d(out_channel, out_channel, kernel_size, padding=padding)
        self.noise2 = equal_lr(NoiseInjection(out_channel))
        self.adain2 = AdaptiveInstanceNorm(out_channel, style_dim)
        self.lrelu2 = nn.LeakyReLU(0.2)

    def forward(self, input, style, noise):
        out = self.conv1(input)
        out = self.noise1(out, noise)
        out = self.lrelu1(out)
        out = self.adain1(out, style)

        out = self.conv2(out)
        out = self.noise2(out, noise)
        out = self.lrelu2(out)
        out = self.adain2(out, style)

        return out


class Generator(nn.Module):
    def __init__(self, code_dim, fused=True):
        super().__init__()

        self.progression = nn.ModuleList([
            StyledConvBlock(512, 512, 3, 1, initial=True),  # 4
            StyledConvBlock(512, 512, 3, 1, upsample=True),  # 8
            StyledConvBlock(512, 512, 3, 1, upsample=True),  # 16
            StyledConvBlock(512, 512, 3, 1, upsample=True),  # 32
            StyledConvBlock(512, 256, 3, 1, upsample=True),  # 64
            StyledConvBlock(256, 128, 3, 1, upsample=True, fused=fused),  # 128
            StyledConvBlock(128, 64, 3, 1, upsample=True, fused=fused),  # 256
            StyledConvBlock(64, 32, 3, 1, upsample=True, fused=fused),  # 512
            StyledConvBlock(32, 16, 3, 1, upsample=True, fused=fused),  # 1024
        ])

        self.to_rgb = nn.ModuleList([
            EqualConv2d(512, 3, 1),
            EqualConv2d(512, 3, 1),
            EqualConv2d(512, 3, 1),
            EqualConv2d(512, 3, 1),
            EqualConv2d(256, 3, 1),
            EqualConv2d(128, 3, 1),
            EqualConv2d(64, 3, 1),
            EqualConv2d(32, 3, 1),
            EqualConv2d(16, 3, 1),
        ])

    def forward(self, style, noise, step=0, alpha=-1):
        # NOTE eliminate style mixing
        out = noise[0]

        if len(style) < 2:
            inject_index = [len(self.progression) + 1]
        else:
            inject_index = random.sample(list(range(step)), len(style) - 1)

        crossover = 0
        for i, (conv, to_rgb) in enumerate(zip(self.progression, self.to_rgb)):
            if crossover < len(inject_index) and i > inject_index(crossover):
                crossover = min(crossover + 1, len(style))
            
            style_step = style[crossover]

            if i > 0 and step > 0:
                out_prev = out
                out = conv(out, style_step, noise[i])
            else:
                out = conv(out, style_step, noise[i])

            if i == step:
                out = to_rgb(out)

                if i > 0 and  0 <= alpha < 1:
                    skip_rgb = self.to_rgb[i - 1](out_prev)
                    skip_rgb = F.interpolate(skip_rgb, scale_factor=2, mode='nearest')
                    out = (1 - alpha) * skip_rgb + alpha * out

                break
        
        return out


class PoseEncoder(nn.Module):
    def __init__(self, code_dim, fused=True):
        super().__init__()

        self.cnn = nn.ModuleList([
            ConvBlock(3, 32, 3, 1, downsample=True, fused=fused), # 512
            ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
            ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 128
            ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
            ConvBlock(256, 512, 3, 1, downsample=True),  # 32
            ConvBlock(512, 512, 3, 1, downsample=True),  # 16
            ConvBlock(512, 512, 3, 1, downsample=True),  # 8
            ConvBlock(512, 512, 3, 1, downsample=True),  # 4
            ConvBlock(512, 512, 3, 1, 4, 0),  # 1
        ])

        self.linear = EqualLinear(512, code_dim)
        # self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, input):

        for conv in self.cnn:
            out = conv(input)

        out = self.linear(out)
        # out = self.lrelu(out)

        return out



class PoseGenerator(nn.Module):
    def __init__(self, code_dim=512):
        super().__init__()
        
        self.style = PoseEncoder(code_dim)
        self.generator = Generator(code_dim)

    def forward(
        self,
        input,
        noise=None,
        step=0,
        alpha=-1,
    ):
        styles = []
        if type(input) not in (list, tuple):
            input = [input]

        for i in input:
            styles.append(self.style(i))

        batch = input[0].shape[0]

        if noise is None:
            noise = []
            for i in range(step + 1):
                size = 4 * 2 ** i
                noise.append(torch.randn(batch, 1, size, size, device=input[0].device))

        return self.generator(styles, noise, step, alpha)


class Discriminator(nn.Module):
    def __init__(self, fused=True, from_rgb_activate=False):
        super().__init__()

        self.progression = nn.ModuleList([
            ConvBlock(16, 32, 3, 1, downsample=True, fused=fused),  # 512
            ConvBlock(32, 64, 3, 1, downsample=True, fused=fused),  # 256
            ConvBlock(64, 128, 3, 1, downsample=True, fused=fused),  # 128
            ConvBlock(128, 256, 3, 1, downsample=True, fused=fused),  # 64
            ConvBlock(256, 512, 3, 1, downsample=True),  # 32
            ConvBlock(512, 512, 3, 1, downsample=True),  # 16
            ConvBlock(512, 512, 3, 1, downsample=True),  # 8
            ConvBlock(512, 512, 3, 1, downsample=True),  # 4
            # minibatch discrimination
            ConvBlock(513, 512, 3, 1, 4, 0),  # 1
        ])

        def make_from_rgb(in_channel, out_channel):
            if from_rgb_activate:
                return nn.Sequential(EqualConv2d(in_channel, out_channel, 1), nn.LeakyReLU(0.2))
            else:
                return EqualConv2d(in_channel, out_channel, 1)

        self.from_rgb = nn.ModuleList([
            make_from_rgb(6, 16),   # 1024
            make_from_rgb(6, 32),   # 512
            make_from_rgb(6, 64),   # 256
            make_from_rgb(6, 128),  # 128
            make_from_rgb(6, 256),  # 64
            make_from_rgb(6, 512),  # 32
            make_from_rgb(6, 512),  # 16
            make_from_rgb(6, 512),  # 8
            make_from_rgb(6, 512),  # 4
        ])

        self.n_layer = len(self.progression)
        self.linear = EqualLinear(512, 1)

    def forward(self, input, step=0, alpha=-1):
        steps_out = []
        for i in range(step, -1, -1):
            index = self.n_layer - i - 1

            if i == step:
                out = self.from_rgb[index](input)
            
            if i == 0:
                out_std = torch.sqrt(out.var(0, unbiased=False) + 1e-8)
                mean_std = out_std.mean()
                mean_std = mean_std.expand(out.size(0), 1, 4, 4)
                out = torch.cat([out, mean_std], 1)

            out = self.progression[index](out)

            if i > 0:
                if i == step and 0 <= alpha <= 1:
                    skip_rgb = F.avg_pool2d(input, 2)
                    skip_rgb = self.from_rgb[index + 1](skip_rgb)
                    out = (1 - alpha) * skip_rgb + alpha * out
        
            if i != 0:
                steps_out.append(out)

        out = out.squeeze(2).squeeze(2)
        out = self.linear(out)
        steps_out.append(out)

        return steps_out


class PoseDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.discriminator = Discriminator()

    def forward(self, input, step=0, alpha=-1):
        
        return self.discriminator(input, step, alpha)


def weights_init(m):
    '''
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # m.weight.data.normal_(0.0, 0.02)
        init.kaiming_normal_(m.weight)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight, mean=1.0, std=0.02)

    if m.bias is not None:
        m.bias.data.fill_(0)
    '''
    pass



def define_G(code_dim=512, gpu_ids=None):
    netG = PoseGenerator(code_dim)
    print(netG)

    if gpu_ids is not None:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    netG.apply(weights_init)

    return netG


def define_D(gpu_ids=None):
    netD = PoseDiscriminator()
    print(netD)

    if gpu_ids is not None:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)

    return netD


# Losses
class GANLoss(nn.Module):
    
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super().__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_tensor = None
        self.fake_label_tensor = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    
    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_tensor is None) or (self.real_label_tensor.numel() != input.numel()))
            if create_label:
                self.real_label_tensor = self.Tensor(input.size()).fill_(self.real_label)
            target_tensor = self.real_label_tensor

        else:
            create_label = ((self.fake_label_tensor is None) or (self.fake_label_tensor.numel() != input.numel()))
            if create_label:
                self.fake_label_tensor = self.Tensor(input.size()).fill_(self.fake_label)
            target_tensor = self.fake_label_tensor

        return target_tensor
        

    def __call__(self, input, target_is_real):
        # last output
        target_tensor = self.get_target_tensor(input[-1], target_is_real)
        return self.loss(input[-1], target_tensor)



class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super().__init__()

        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):

        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        
        return loss


class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super().__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]

        return out
