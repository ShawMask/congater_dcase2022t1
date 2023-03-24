"""
Most of this code comes from the timm  library.
We tried to disentangle from the timm library version.

Adapted from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py

"""
import math
import logging
import warnings
from functools import partial
from collections import OrderedDict
from copy import deepcopy
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers.helpers import to_2tuple

from models.passt.helpers.vit_helpers import update_default_cfg_and_kwargs, DropPath, trunc_normal_, build_model_with_cfg
from models.passt.passt import PaSST

_logger = logging.getLogger()

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5)
IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
    ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz'),
    'vit_base_patch16_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz'),

    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),
    # PaSST
    'passt_s_swa_p16_128_ap476': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.1-audioset/passt-s-f128-p16-s10-ap.476-swa.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 998), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=527),
    'passt_s_swa_p16_128_ap4761': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.2-audioset/passt-s-f128-p16-s10-ap.4761-swa.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 998), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=527),
    'passt_s_p16_128_ap472': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.2-audioset/passt-s-f128-p16-s10-ap.472.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 998), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=527),
    'passt_s_p16_s16_128_ap468': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.2-audioset/passt-s-f128-p16-s16-ap.468.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 998), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=527),
    'passt_s_swa_p16_s16_128_ap473': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.2-audioset/passt-s-f128-p16-s16-ap.473-swa.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 998), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=527),
    'passt_s_swa_p16_s14_128_ap471': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.2-audioset/passt-s-f128-p16-s14-ap.471-swa.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 998), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=527),
    'passt_s_p16_s14_128_ap469': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.2-audioset/passt-s-f128-p16-s14-ap.469.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 998), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=527),
    'passt_s_swa_p16_s12_128_ap473': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.2-audioset/passt-s-f128-p16-s12-ap.473-swa.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 998), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=527),
    'passt_s_p16_s12_128_ap470': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.2-audioset/passt-s-f128-p16-s12-ap.470.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 998), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=527),
    'passt_s_swa_f128_stfthop100_p16_s10_ap473': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.3-audioset/passt-s-f128-stfthop100-p16-s10-ap.473-swa.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 3200), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=527),
    'passt_s_swa_f128_stfthop160_p16_s10_ap473': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.3-audioset/passt-s-f128-stfthop160-p16-s10-ap.473-swa.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 2000), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=527),
    'passt-s-f128-20sec-p16-s10-ap474-swa': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.3-audioset/passt-s-f128-20sec-p16-s10-ap.474-swa.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 2000), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=527),
    'passt-s-f128-30sec-p16-s10-ap473-swa': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.3-audioset/passt-s-f128-30sec-p16-s10-ap.473-swa.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 3000), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=527),
    'openmic2008_passt_u_f128_p16_s10_ap85_swa': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.4-openmic/openmic2008.passt-u-f128-p16-s10-ap.85-swa.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 3200), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=20),
    'openmic2008_passt_u_f128_p16_s10_ap85  ': _cfg(
        url='https://github.com/kkoutini/PaSST/releases/download/v0.0.4-openmic/openmic2008.passt-u-f128-p16-s10-ap.85.pt',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(1, 128, 2000), crop_pct=1.0,
        classifier=('head.1', 'head_dist'), num_classes=20),
}

# Custom Sigmoid Function designed to transform from sigmoid to uniform 1
class Tsigmoid(nn.Module):
    def __init__(self):
        super(Tsigmoid, self).__init__()

    def forward(self, x, t):
        scale = 2. - t if t <= 1 else 1.
        out = scale / (1 + torch.exp(-t * x))
        out[out > 1] = 1
        return out

class Lsigmoid(nn.Module):
    def __init__(self):
        super(Lsigmoid, self).__init__()

    def forward(self, x, t):
        omega = torch.log2(torch.tensor(t+1))  # 1<t<2
        out = 1 - omega / (1 + torch.exp(x))
        return out

# Gating Mechanism Can be Linear or none linear transform
class GateLayer(nn.Module):
    def __init__(self, embed_size, num_layers=1, gate_activation="T"):
        super(GateLayer, self).__init__()
        self.nonlinear = nn.ModuleList()
        self.num_layers = num_layers
        for i in range(num_layers):
            if i < (num_layers - 1):
                self.nonlinear.append(nn.Sequential(nn.Linear(embed_size, embed_size), nn.ReLU()))
            else:
                self.nonlinear.append(nn.Linear(embed_size, embed_size))

        if gate_activation == "L":
            self.activation = Lsigmoid()
        else:
            self.activation = Tsigmoid()

    def forward(self, embed, temp):
        for i in range(self.num_layers):
            embed = self.nonlinear[i](embed)

        return self.activation(embed, temp)


def load_passt(arch="passt_s_swa_p16_128_ap476", fstride=10, tstride=10, n_classes=10, input_fdim=128, input_tdim=998,
               u_patchout=0, s_patchout_t=0, s_patchout_f=0, model_number=None, pretrained=True,
               in_channels=1,):

    model = get_model(arch=arch, fstride=fstride, pretrained=pretrained, tstride=tstride, in_channels=in_channels,
                      n_classes=n_classes, input_fdim=input_fdim,
                      input_tdim=input_tdim, u_patchout=u_patchout, s_patchout_t=s_patchout_t,
                      s_patchout_f=s_patchout_f)

    if model_number:
        path_ = os.path.join('teacher_models', f"passt_{model_number}.pt")
        print(path_)
        print("\n\n Loaded Teacher from ", f"{path_}", "\n\n")
        check_point = torch.load(path_)
        model.load_state_dict(check_point)
    return model

def adapt_input_conv(in_chans, conv_weight):
    conv_type = conv_weight.dtype
    conv_weight = conv_weight.float()  # Some weights are in torch.half, ensure it's float for sum on CPU
    O, I, J, K = conv_weight.shape
    if in_chans == 1:
        if I > 3:
            assert conv_weight.shape[1] % 3 == 0
            # For models with space2depth stems
            conv_weight = conv_weight.reshape(O, I // 3, 3, J, K)
            conv_weight = conv_weight.sum(dim=2, keepdim=False)
        else:
            conv_weight = conv_weight.sum(dim=1, keepdim=True)
    elif in_chans != 3:
        if I != 3:
            raise NotImplementedError('Weight format not supported by conversion.')
        else:
            # NOTE this strategy should be better than random init, but there could be other combinations of
            # the original RGB input layer weights that'd work better for specific cases.
            repeat = int(math.ceil(in_chans / 3))
            conv_weight = conv_weight.repeat(1, repeat, 1, 1)[:, :in_chans, :, :]
            conv_weight *= (3 / float(in_chans))
    conv_weight = conv_weight.to(conv_type)
    return conv_weight


first_RUN = True


class ConGater(nn.Module):
    """

    Based on the implementation of Vision Transformer in timm library.
     Take a look at the get_model function, adapting the weights of pretrained imagenet models.

    """

    def __init__(self, arch="passt_s_swa_p16_128_ap476", fstride=10, tstride=10, n_classes=10, input_fdim=128,
                 input_tdim=998, u_patchout=0, s_patchout_t=0, s_patchout_f=0, model_number=None, target_domain=None,
                 gate_activation="T", congater_loc=None, num_gate_layers=1):
        """
        Args:
            u_patchout: Unstructured Patchout integer, number of items to be removed from the final sequence
            s_patchout_t: structured Patchout time integer, number of columns to be removed from the patches grid
            s_patchout_f: structured Patchout Frequency integer, number of rows to be removed from the patches grid
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()

        self.model_number = model_number
        self.target_domain = target_domain
        self.num_gate_layers = num_gate_layers
        self.gate_activation = gate_activation if self.target_domain else "N"
        self.passt = load_passt(arch=arch, fstride=fstride, tstride=tstride, n_classes=n_classes, input_fdim=input_fdim,
                                input_tdim=input_tdim, u_patchout=u_patchout, s_patchout_t=s_patchout_t,
                                s_patchout_f=s_patchout_f, model_number=model_number)
        self.domain = self.target_domain.split(" ") if self.target_domain else None
        self.embed_dim = self.passt.embed_dim
        self.congater_loc = congater_loc
        if self.target_domain:
            self.congater = nn.ModuleDict()
            for i in self.domain:
                self.congater[i] = nn.ModuleList()
                if self.congater_loc == "all":
                    for _ in range(len(self.passt.blocks)):
                        self.congater[i].append(GateLayer(self.passt.embed_dim, num_layers=self.num_gate_layers,
                                                              gate_activation=self.gate_activation))
                        with torch.no_grad():
                            self.congater[i][-1].nonlinear[0].weight = \
                                nn.Parameter(torch.zeros_like(self.congater[i][-1].nonlinear[0].weight),
                                             requires_grad=True)
                            self.congater[i][-1].nonlinear[0].bias = \
                                nn.Parameter(torch.ones_like(self.congater[i][-1].nonlinear[0].bias) * 6,
                                             requires_grad=True)

                elif self.congater_loc == "last":
                    self.congater[i] = nn.ModuleList()
                    self.congater[i].append(GateLayer(self.passt.embed_dim, num_layers=self.num_gate_layers,
                                                      gate_activation=self.gate_activation))
                    with torch.no_grad():
                        self.congater[i][-1].nonlinear[0].weight = \
                            nn.Parameter(torch.zeros_like(self.congater[i][-1].nonlinear[0].weight), requires_grad=True)
                        self.congater[i][-1].nonlinear[0].bias = \
                            nn.Parameter(torch.ones_like(self.congater[i][-1].nonlinear[0].bias)*6, requires_grad=True)
        else:
            self.congater = None

    def condition_(self, param, text):
        if '+' in param:
            a = param.split('+')
            b = []
            for t in a:
                b.append(t.split(' '))
        else:
            b = param.split(' ')
        if type(b[0]) == str:
            if all(word in text for word in b):
                return True
            else:
                return False
        if len(b) == 2:
            if all(word in text for word in b[0]) or all(word in text for word in b[1]):
                return True
            else:
                return False
        if len(b) == 3:
            if all(word in text for word in b[0]) or all(word in text for word in b[1]) or all(
                    word in text for word in b[2]):
                return True
            else:
                return False

    def set_trainable_parameters(self, trainable_param):
        if trainable_param == 'none':
            for name, param in self.named_parameters():
                if 'head' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif trainable_param == 'all':
            for n, param in self.named_parameters():
                param.requires_grad = True
        else:
            for name, param in self.named_parameters():
                if self.condition_(trainable_param, name):
                    param.requires_grad = True
                # elif 'passt.norm' in name:
                #     param.requires_grad = True
                else:
                    param.requires_grad = False





    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.new_pos_embed, std=.02)
        trunc_normal_(self.freq_new_pos_embed, std=.02)
        trunc_normal_(self.time_new_pos_embed, std=.02)
        if self.dist_token is not None:
            trunc_normal_(self.dist_token, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            raise RuntimeError("Not supported yet")
        else:
            trunc_normal_(self.cls_token, std=.02)
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'new_pos_embed', 'freq_new_pos_embed', 'time_new_pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()


    def forward_gate(self, x, temp, layer):
        x_gate = []
        # temp = {"device": temp[0], "location": temp[1]}
        for name, t in temp.items():
                x_gate.append(self.congater[name][layer](x, t))

        if len(x_gate) > 1:
            x_gate = [x_gate[i]*x_gate[i-1] for i in range(1, len(x_gate))][0]
        else:
            x_gate = x_gate[0]
        return x_gate

    def forward_features(self, x, temp):
        global first_RUN  # not jit friendly? use trace instead
        x = self.passt.patch_embed(x)  # [b, e, f, t]
        B_dim, E_dim, F_dim, T_dim = x.shape  # slow
        if first_RUN: print(" patch_embed : ", x.shape)
        # Adding Time/Freq information
        if first_RUN: print(" self.time_new_pos_embed.shape", self.passt.time_new_pos_embed.shape)
        time_new_pos_embed = self.passt.time_new_pos_embed
        if x.shape[-1] < time_new_pos_embed.shape[-1]:
            if self.passt.training:
                toffset = torch.randint(1 + time_new_pos_embed.shape[-1] - x.shape[-1], (1,)).item()
                if first_RUN: print(f" CUT with randomoffset={toffset} time_new_pos_embed.shape",
                                    time_new_pos_embed.shape)
                time_new_pos_embed = time_new_pos_embed[:, :, :, toffset:toffset + x.shape[-1]]
            else:
                time_new_pos_embed = time_new_pos_embed[:, :, :, :x.shape[-1]]
            if first_RUN: print(" CUT time_new_pos_embed.shape", time_new_pos_embed.shape)
        else:
            warnings.warn(
                f"the patches shape:{x.shape} are larger than the expected time encodings {time_new_pos_embed.shape}, x will be cut")
            x = x[:, :, :, :time_new_pos_embed.shape[-1]]
        x = x + time_new_pos_embed
        if first_RUN: print(" self.freq_new_pos_embed.shape", self.passt.freq_new_pos_embed.shape)
        x = x + self.passt.freq_new_pos_embed

        # Structured Patchout https://arxiv.org/abs/2110.05069 Section 2.2
        if self.passt.training and self.passt.s_patchout_t:
            if first_RUN: print(f"X Before time Patchout of {self.passt.s_patchout_t} ", x.size())
            # ([1, 768, 1, 82])
            random_indices = torch.randperm(T_dim)[:T_dim - self.passt.s_patchout_t].sort().values
            x = x[:, :, :, random_indices]
            if first_RUN: print("X after time Patchout", x.size())
        if self.passt.training and self.passt.s_patchout_f:
            if first_RUN: print(f"X Before Freq Patchout of {self.passt.s_patchout_f} ", x.size())
            # [1, 768, 12, 1]
            random_indices = torch.randperm(F_dim)[:F_dim - self.passt.s_patchout_f].sort().values
            x = x[:, :, random_indices, :]
            if first_RUN: print(" \n X after freq Patchout: ", x.size())
        ###
        # Flatten the sequence
        x = x.flatten(2).transpose(1, 2)
        # Unstructured Patchout
        if first_RUN: print("X flattened", x.size())
        if self.passt.training and self.passt.u_patchout:
            seq_len = x.shape[1]
            random_indices = torch.randperm(seq_len)[:seq_len - self.passt.u_patchout].sort().values
            x = x[:, random_indices, :]
            if first_RUN: print("X After Unstructured Patchout", x.size())
        ####
        # Add the C/D tokens
        if first_RUN: print(" self.new_pos_embed.shape", self.passt.new_pos_embed.shape)
        cls_tokens = self.passt.cls_token.expand(B_dim, -1, -1) + self.passt.new_pos_embed[:, :1, :]
        if first_RUN: print(" self.cls_tokens.shape", cls_tokens.shape)
        if self.passt.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)
        else:
            dist_token = self.passt.dist_token.expand(B_dim, -1, -1) + self.passt.new_pos_embed[:, 1:, :]
            if first_RUN: print(" self.dist_token.shape", dist_token.shape)
            x = torch.cat((cls_tokens, dist_token, x), dim=1)

        if first_RUN: print(" final sequence x", x.shape)
        x = self.passt.pos_drop(x)
        if self.target_domain: # Self Gate of the ConGater
            if self.congater_loc == "all":
                for cnt, block in enumerate(self.passt.blocks):
                    x = block(x) * self.forward_gate(x, temp, cnt)
            elif self.congater_loc == "last":
                x = self.passt.blocks(x)
                x = x*self.forward_gate(x, temp, -1)
        else:
            x = self.passt.blocks(x)
        if first_RUN: print(f" after {len(self.passt.blocks)} atten blocks x", x.shape)
        x = self.passt.norm(x)
        if self.passt.dist_token is None:
            return self.passt.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x, temp):
        global first_RUN
        if first_RUN: print("x", x.size())

        x = self.forward_features(x, temp)

        if self.passt.head_dist is not None:
            features = (x[0] + x[1]) / 2
            if first_RUN: print("forward_features", features.size())
            x = self.passt.head(features)
            if first_RUN: print("head", x.size())
            first_RUN = False
            return x, features
        else:
            features = x
            if first_RUN: print("forward_features", features.size())
            x = self.passt.head(x)
        if first_RUN: print("head", x.size())
        first_RUN = False
        return x, features


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=(), mode='bicubic'):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s with %s cls/dis tokens', posemb.shape, posemb_new.shape,
                 num_tokens)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=mode, align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def adapt_image_pos_embed_to_passt(posemb, num_tokens=1, gs_new=(), mode='bicubic'):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s with %s cls/dis tokens', posemb.shape, gs_new,
                 num_tokens)
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))

    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode=mode, align_corners=False)
    freq_new_pos_embed = posemb_grid.mean(dim=3, keepdim=True)
    time_new_pos_embed = posemb_grid.mean(dim=2, keepdim=True)
    _logger.info('New Position cls/dstl embedding %s', posemb_tok.shape)
    _logger.info('New FREQ Position embedding %s', freq_new_pos_embed.shape)
    _logger.info('New TIME Position embedding %s', time_new_pos_embed.shape)
    return posemb_tok, freq_new_pos_embed, time_new_pos_embed


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    state_dict = {k: v for k, v in state_dict.items()}
    if "time_new_pos_embed" not in state_dict:
        # we are working with ImageNet model
        _logger.info("Adapting pos embedding from ImageNet pretrained model to PaSST.")
        v = state_dict.pop("pos_embed")
        new_pos_embed, freq_new_pos_embed, time_new_pos_embed = adapt_image_pos_embed_to_passt(
            v, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        state_dict["new_pos_embed"] = new_pos_embed
        state_dict["freq_new_pos_embed"] = freq_new_pos_embed
        state_dict["time_new_pos_embed"] = time_new_pos_embed

    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # this should never occur
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_vision_transformer(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    default_num_classes = default_cfg['num_classes']
    num_classes = kwargs.get('num_classes', default_num_classes)
    repr_size = kwargs.pop('representation_size', None)
    if repr_size is not None and num_classes != default_num_classes:
        # Remove representation layer if fine-tuning. This may not always be the desired action,
        # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
        _logger.warning("Removing representation layer for fine-tuning.")
        repr_size = None

    model = build_model_with_cfg(
        PaSST, variant, pretrained,
        default_cfg=default_cfg,
        representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


def vit_huge_patch14_224_in21k(pretrained=False, **kwargs):
    """ ViT-Huge model (ViT-H/14) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-21k weights @ 224x224, source https://github.com/google-research/vision_transformer.
    NOTE: this model has a representation layer but the 21k classifier head is zero'd out in original weights
    """
    model_kwargs = dict(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16, representation_size=1280, **kwargs)
    model = _create_vision_transformer('vit_huge_patch14_224_in21k', pretrained=pretrained, **model_kwargs)
    return model


def deit_base_distilled_patch16_384(pretrained=False, **kwargs):
    """ DeiT-base distilled model @ 384x384 from paper (https://arxiv.org/abs/2012.12877).
    ImageNet-1k weights from https://github.com/facebookresearch/deit.
    """
    print("\n\n Loading DEIT BASE 384\n\n")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'deit_base_distilled_patch16_384', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


def passt_s_swa_p16_128_ap476(pretrained=False, **kwargs):
    """ PaSST pre-trained on AudioSet
    """
    print("\n\n Loading PaSST pre-trained on AudioSet Patch 16 stride 10 structured patchout mAP=476 SWA \n\n")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if model_kwargs.get("stride") != (10, 10):
        warnings.warn(
            f"This model was pre-trained with strides {(10, 10)}, but now you set (fstride,tstride) to {model_kwargs.get('stride')}.")
    model = _create_vision_transformer(
        'passt_s_swa_p16_128_ap476', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


def passt_s_swa_p16_128_ap4761(pretrained=False, **kwargs):
    """ PaSST pre-trained on AudioSet
    """
    print("\n\n Loading PaSST pre-trained on AudioSet Patch 16 stride 10 structured patchout mAP=4763 SWA \n\n")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if model_kwargs.get("stride") != (10, 10):
        warnings.warn(
            f"This model was pre-trained with strides {(10, 10)}, but now you set (fstride,tstride) to {model_kwargs.get('stride')}.")
    model = _create_vision_transformer(
        'passt_s_swa_p16_128_ap4761', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


def passt_s_p16_128_ap472(pretrained=False, **kwargs):
    """ PaSST pre-trained on AudioSet
    """
    print("\n\n Loading PaSST pre-trained on AudioSet Patch 16 stride 10 structured patchout mAP=472 \n\n")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if model_kwargs.get("stride") != (10, 10):
        warnings.warn(
            f"This model was pre-trained with strides {(10, 10)}, but now you set (fstride,tstride) to {model_kwargs.get('stride')}.")
    model = _create_vision_transformer(
        'passt_s_p16_128_ap472', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


def passt_s_p16_s12_128_ap470(pretrained=False, **kwargs):
    """ PaSST pre-trained on AudioSet
    """
    print("\n\n Loading PaSST pre-trained on AudioSet Patch 16 stride 12 structured patchout mAP=472 \n\n")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if model_kwargs.get("stride") != (12, 12):
        warnings.warn(
            f"This model was pre-trained with strides {(12, 12)}, but now you set (fstride,tstride) to {model_kwargs.get('stride')}.")
    model = _create_vision_transformer(
        'passt_s_p16_s12_128_ap470', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


def passt_s_f128_20sec_p16_s10_ap474_swa(pretrained=False, **kwargs):
    print("\n\n Loading PASST TRAINED ON AUDISET with 20 Second time encodings, with STFT hop of 160 \n\n")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'passt-s-f128-20sec-p16-s10-ap474-swa', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


def passt_s_f128_30sec_p16_s10_ap473_swa(pretrained=False, **kwargs):
    print("\n\n Loading PASST TRAINED ON AUDISET with 30 Second time encodings, with STFT hop of 160 \n\n")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(
        'passt-s-f128-30sec-p16-s10-ap473-swa', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


def passt_s_swa_p16_s12_128_ap473(pretrained=False, **kwargs):
    """ PaSST pre-trained on AudioSet
    """
    print("\n\n Loading PaSST pre-trained on AudioSet Patch 16 stride 12 structured patchout mAP=472 \n\n")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if model_kwargs.get("stride") != (12, 12):
        warnings.warn(
            f"This model was pre-trained with strides {(12, 12)}, but now you set (fstride,tstride) to {model_kwargs.get('stride')}.")
    model = _create_vision_transformer(
        'passt_s_swa_p16_s12_128_ap473', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


def passt_s_p16_s14_128_ap469(pretrained=False, **kwargs):
    """ PaSST pre-trained on AudioSet
    """
    print("\n\n Loading PaSST pre-trained on AudioSet Patch 16 stride 14 structured patchout mAP=472 \n\n")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if model_kwargs.get("stride") != (14, 14):
        warnings.warn(
            f"This model was pre-trained with strides {(14, 14)}, but now you set (fstride,tstride) to {model_kwargs.get('stride')}.")
    model = _create_vision_transformer(
        'passt_s_p16_s14_128_ap469', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


def passt_s_swa_p16_s14_128_ap471(pretrained=False, **kwargs):
    """ PaSST pre-trained on AudioSet
    """
    print("\n\n Loading PaSST pre-trained on AudioSet Patch 16 stride 14 structured patchout mAP=472 \n\n")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if model_kwargs.get("stride") != (14, 14):
        warnings.warn(
            f"This model was pre-trained with strides {(14, 14)}, but now you set (fstride,tstride) to {model_kwargs.get('stride')}.")
    model = _create_vision_transformer(
        'passt_s_swa_p16_s14_128_ap471', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


def passt_s_swa_p16_s16_128_ap473(pretrained=False, **kwargs):
    """ PaSST pre-trained on AudioSet
    """
    print("\n\n Loading PaSST pre-trained on AudioSet Patch 16 stride 16 structured patchout mAP=472 \n\n")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if model_kwargs.get("stride") != (16, 16):
        warnings.warn(
            f"This model was pre-trained with strides {(16, 16)}, but now you set (fstride,tstride) to {model_kwargs.get('stride')}.")
    model = _create_vision_transformer(
        'passt_s_swa_p16_s16_128_ap473', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


def passt_s_p16_s16_128_ap468(pretrained=False, **kwargs):
    """ PaSST pre-trained on AudioSet
    """
    print("\n\n Loading PaSST pre-trained on AudioSet Patch 16 stride 16 structured patchout mAP=472 \n\n")
    model_kwargs = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    if model_kwargs.get("stride") != (16, 16):
        warnings.warn(
            f"This model was pre-trained with strides {(16, 16)}, but now you set (fstride,tstride) to {model_kwargs.get('stride')}.")
    model = _create_vision_transformer(
        'passt_s_p16_s16_128_ap468', pretrained=pretrained, distilled=True, **model_kwargs)
    return model


from ba3l.ingredients.ingredient import Ingredient

model_ing = Ingredient("congater")

model_ing.add_config(instance_cmd="load_congater")


@model_ing.command
def fix_embedding_layer(model, embed="default"):
    if embed == "default":
        return model
    if embed == "overlap":
        model.patch_embed = PatchEmbedAdaptiveMean(replace=model.patch_embed)
    if embed == "am_keepconv":
        model.patch_embed = PatchEmbedAdaptiveMeanKeepConv(replace=model.patch_embed)
    return model

@model_ing.command
def lighten_model(model, cut_depth=0):
    if cut_depth == 0:
        return model
    if cut_depth:
        if cut_depth < 0:
            print(f"\n Reducing model depth by removing every  {-cut_depth} layer \n\n")
        else:
            print(f"\n Reducing model depth by {cut_depth} \n\n")
            if len(model.blocks) < cut_depth + 2:
                raise ValueError(f"Cut depth a VIT with {len(model.blocks)} "
                                 f"layers should be between 1 and {len(model.blocks) - 2}")
        print(f"\n Before Cutting it was  {len(model.blocks)} \n\n")

        old_blocks = list(model.blocks.children())
        if cut_depth < 0:
            print(f"cut_depth={cut_depth}")
            old_blocks = [old_blocks[0]] + old_blocks[1:-1:-cut_depth] + [old_blocks[-1]]
        else:
            old_blocks = [old_blocks[0]] + old_blocks[cut_depth + 1:]
        model.blocks = nn.Sequential(*old_blocks)
        print(f"\n Atfer Cutting it is  {len(model.blocks)} \n\n")
    return model


@model_ing.command
def get_model(arch="passt_s_swa_p16_128_ap476", pretrained=True, n_classes=527, in_channels=1, fstride=10,
              tstride=10, input_fdim=128, input_tdim=998, u_patchout=0, s_patchout_t=0, s_patchout_f=0,
              ):
    """
    :param arch: Base ViT or Deit architecture
    :param pretrained: use pretrained model on imagenet
    :param n_classes: number of classes
    :param in_channels: number of input channels: 1 for mono
    :param fstride: the patches stride over frequency.
    :param tstride: the patches stride over time.
    :param input_fdim: the expected input frequency bins.
    :param input_tdim: the expected input time bins.
    :param u_patchout: number of input patches to drop in Unstructured Patchout as defined in https://arxiv.org/abs/2110.05069
    :param s_patchout_t: number of input time frames to drop Structured Patchout as defined in https://arxiv.org/abs/2110.05069
    :param s_patchout_f:  number of input frequency bins to drop Structured Patchout as defined in https://arxiv.org/abs/2110.05069
    :param audioset_pretrain: use pretrained models on Audioset.
    :return:

    """
    model_func = None
    input_size = (input_fdim, input_tdim)
    stride = (fstride, tstride)
    if arch == "passt_deit_bd_p16_384":  # base deit
        model_func = deit_base_distilled_patch16_384
    elif arch == "passt_s_swa_p16_128_ap476":  # pretrained
        model_func = passt_s_swa_p16_128_ap476
    elif arch == "passt_s_swa_p16_128_ap4761":
        model_func = passt_s_swa_p16_128_ap4761
    elif arch == "passt_s_p16_128_ap472":
        model_func = passt_s_p16_128_ap472
    elif arch == "passt_s_p16_s16_128_ap468":
        model_func = passt_s_p16_s16_128_ap468
    elif arch == "passt_s_swa_p16_s16_128_ap473":
        model_func = passt_s_swa_p16_s16_128_ap473
    elif arch == "passt_s_swa_p16_s14_128_ap471":
        model_func = passt_s_swa_p16_s14_128_ap471
    elif arch == "passt_s_p16_s14_128_ap469":
        model_func = passt_s_p16_s14_128_ap469
    elif arch == "passt_s_swa_p16_s12_128_ap473":
        model_func = passt_s_swa_p16_s12_128_ap473
    elif arch == "passt_s_p16_s12_128_ap470":
        model_func = passt_s_p16_s12_128_ap470
    elif arch == "passt_s_f128_20sec_p16_s10_ap474":
        model_func = passt_s_f128_20sec_p16_s10_ap474_swa
    elif arch == "passt_s_f128_30sec_p16_s10_ap473":
        model_func = passt_s_f128_30sec_p16_s10_ap473_swa

    if model_func is None:
        raise RuntimeError(f"Unknown model {arch}")
    model = model_func(pretrained=pretrained, num_classes=n_classes, in_chans=in_channels,
                       img_size=input_size, stride=stride, u_patchout=u_patchout,
                       s_patchout_t=s_patchout_t, s_patchout_f=s_patchout_f)
    model = fix_embedding_layer(model)
    model = lighten_model(model)
    # print(model)
    return model


@model_ing.command
def load_congater(arch="passt_s_swa_p16_128_ap476", n_classes=527, fstride=10,
                  tstride=10, input_fdim=128, input_tdim=998, u_patchout=0, s_patchout_t=0, s_patchout_f=0,
                  model_number=None, target_domain=None, gate_activation="T", congater_loc="all", num_gate_layers=1):
    model = ConGater(arch=arch,n_classes=n_classes, fstride=fstride,
                     tstride=tstride, input_fdim=input_fdim, input_tdim=input_tdim, u_patchout=u_patchout,
                     s_patchout_t=s_patchout_t, s_patchout_f=s_patchout_f, model_number=model_number,
                     target_domain=target_domain, gate_activation=gate_activation, congater_loc=congater_loc,
                     num_gate_layers=num_gate_layers)
    print(model)
    return model


class EnsembelerModel(nn.Module):
    def __init__(self, models):
        super(EnsembelerModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x):
        # ModuleList can act as an iterable, or be indexed using ints
        all_out = None
        for i, m in enumerate(self.models):
            out, _ = m(x)
            if all_out is None:
                all_out = out
            else:
                all_out = out + all_out
        all_out = all_out / len(self.models)
        return all_out, all_out

@model_ing.command
def get_teacher_avg_ensemble(teachers_list=[], teachers_path="teacher_models"):
    models_list = [get_model(arch="passt_s_swa_p16_128_ap476", fstride=10,
                             tstride=10, n_classes=10) for _ in teachers_list]

    for i, tid in enumerate(teachers_list):
        ckpt = torch.load(f"{teachers_path}/passt_{tid}.pt")
        print("\n\n Loaded Teacher from ", f"{teachers_path}/passt_{tid}.pt", "\n\n")
        models_list[i].load_state_dict(ckpt)
    model = EnsembelerModel(models_list)
    print(model)
    return model


@model_ing.command
def get_ensemble_model(arch_list=[]):
    # arch_list = [(passt_s_swa_p16_128_ap476,fstride,tstride)]
    models_list = [get_model(arch=arch, fstride=fstride, tstride=tstride) for arch, fstride, tstride in arch_list]
    model = EnsembelerModel(models_list)
    print(model)
    return model
