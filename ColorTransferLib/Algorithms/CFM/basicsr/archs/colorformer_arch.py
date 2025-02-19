import torch
import torch.nn as nn

from ColorTransferLib.Algorithms.CFM.basicsr.archs.colorformer_arch_util import CustomPixelShuffle_ICNR, Encoder, UnetBlockWide, NormType, ResBlock, custom_conv_layer
from ColorTransferLib.Algorithms.CFM.basicsr.utils.registry import ARCH_REGISTRY
import math
import numpy as np
import einops
import torch.nn.functional as F


class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fp16_enabled = False
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)

        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ColorMemory(nn.Module):

    def __init__(self,
                 input_dim,
                 embed_dim=768,
                 n_color=512,
                 color_embed=256,
                 color_centers_path='pretrain/color_embed_10000.npy',
                 semantic_centers_path='pretrain/semantic_embed_10000.npy'):
        super().__init__()
        color_centers = np.load(color_centers_path)
        color_centers = color_centers.astype('int')
        color_centers = (color_centers + 128).clip(0, 255)
        self.color_center1 = nn.Parameter(torch.from_numpy(color_centers[0, :, :]), requires_grad=False)
        self.color_center2 = nn.Parameter(torch.from_numpy(color_centers[1, :, :]), requires_grad=False)
        self.color_center3 = nn.Parameter(torch.from_numpy(color_centers[2, :, :]), requires_grad=False)
        self.color_center4 = nn.Parameter(torch.from_numpy(color_centers[3, :, :]), requires_grad=False)

        self.a_embed = nn.Embedding(256, color_embed)
        self.b_embed = nn.Embedding(256, color_embed)

        self.color_embed1 = nn.Linear(color_embed * 2, color_embed)
        self.color_embed2 = nn.Linear(color_embed * 2, color_embed)
        self.color_embed3 = nn.Linear(color_embed * 2, color_embed)
        self.color_embed4 = nn.Linear(color_embed * 2, color_embed)

        semantic_centers = np.load(semantic_centers_path)
        self.semantic_centers = nn.Parameter(torch.from_numpy(semantic_centers), requires_grad=False)

        self.semantic_embed = nn.Linear(semantic_centers.shape[-1], embed_dim)

        self.proj_q = nn.Linear(input_dim, embed_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(color_embed + input_dim)
        self.norm3 = nn.LayerNorm(input_dim + color_embed)
        self.mlp = Mlp(in_features=color_embed + input_dim, out_features=input_dim + color_embed)

        self.last_conv = nn.Conv2d(input_dim + color_embed, input_dim, kernel_size=1)

    def ab2embed(self, color_center):
        a_embed = self.a_embed(color_center[:, 0])
        b_embed = self.b_embed(color_center[:, 1])
        return torch.cat([a_embed, b_embed], dim=-1)

    def forward(self, x, cls):
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)
        x = x.view(b, -1, c)
        q = self.proj_q(self.norm1(x))

        semantic_embed = self.semantic_embed(self.semantic_centers).repeat(x.size(0), 1, 1)
        ab_embed1 = self.ab2embed(self.color_center1)
        ab_embed2 = self.ab2embed(self.color_center2)
        ab_embed3 = self.ab2embed(self.color_center3)
        ab_embed4 = self.ab2embed(self.color_center4)

        color_embed = self.color_embed1(ab_embed1).repeat(x.size(0), 1, 1)*(cls[:,0]).unsqueeze(1).unsqueeze(2)+\
        self.color_embed2(ab_embed2).repeat(x.size(0), 1, 1)*(cls[:,1]).unsqueeze(1).unsqueeze(2)+\
            self.color_embed3(ab_embed3).repeat(x.size(0), 1, 1)*(cls[:,2]).unsqueeze(1).unsqueeze(2)+\
                self.color_embed4(ab_embed4).repeat(x.size(0), 1, 1)*(cls[:,3]).unsqueeze(1).unsqueeze(2)

        k = einops.rearrange(semantic_embed, "n c1 c2 -> n c2 c1")
        v = color_embed
        attn = torch.bmm(q, k)
        attn = F.softmax(attn, dim=-1)
        color_prior = torch.bmm(attn, v)
        x = torch.cat([x, color_prior], dim=2)

        x = self.norm2(x)

        x = x + self.mlp(x)
        x = self.norm3(x)

        x = x.view(b, h, w, -1).permute(0, 3, 1, 2)
        x = x.contiguous()
        return self.last_conv(x)


class TRDecoder(nn.Module):

    def __init__(self,
                 hooks,
                 nf=512,
                 input_channel=3,
                 output_channel=3,
                 ab_segment_classes=65,
                 blur=True,
                 last_norm='Weight',
                 color_centers_path='pretrain/color_embed_10000.npy',
                 semantic_centers_path='pretrain/semantic_embed_10000.npy'):
        super().__init__()
        self.hooks = hooks
        self.nf = nf
        self.blur = blur
        self.last_norm = getattr(NormType, last_norm)
        self.color_centers_path = color_centers_path
        self.semantic_centers_path = semantic_centers_path

        self.layers = self.make_layers()
        embed_dim = nf // 2


        self.last_shuf = CustomPixelShuffle_ICNR(
            embed_dim, embed_dim, blur=self.blur, norm_type=self.last_norm, scale=4)

    def forward(self, x, cls):
        encode_feat = self.hooks[-1].feature
        bs, L, num_feat = encode_feat.shape

        encode_feat = encode_feat.view(bs, int(math.sqrt(L)), int(math.sqrt(L)), num_feat)
        encode_feat = encode_feat.permute(0, 3, 1, 2)
        out = self.layers[0](encode_feat)
        out = self.layers[1](out)
        out = self.layers[2](out)
        out = self.layers[3](out, cls)

        out = self.last_shuf(out)
        return out

    def make_layers(self):
        decoder_layers = []

        e_in_c = self.hooks[-1].feature.shape[2]
        in_c = e_in_c

        out_c = self.nf
        setup_hooks = self.hooks[-2::-1]
        for layer_index, hook in enumerate(setup_hooks):
            feature_c = hook.feature.shape[2]
            if layer_index == len(setup_hooks) - 1:
                out_c = out_c // 2
            decoder_layers.append(
                UnetBlockWide(
                    in_c, feature_c, out_c, hook, blur=self.blur, self_attention=False, norm_type=NormType.Spectral))
            in_c = out_c
        decoder_layers.append(
            ColorMemory(
                input_dim=in_c,
                embed_dim=512,
                n_color=512,
                color_embed=256,
                color_centers_path=self.color_centers_path,
                semantic_centers_path=self.semantic_centers_path))
        return nn.Sequential(*decoder_layers)


@ARCH_REGISTRY.register()
class ColorFormer(nn.Module):

    def __init__(self,
                 encoder_name,
                 pretrained_path='pretrain/GLH.pth',
                 num_input_channels=3,
                 input_size=(256, 256),
                 nf=512,
                 num_output_channels=3,
                 ab_segment_classes=65,
                 last_norm='Weight',
                 do_normalize=True,
                 color_centers_path='pretrain/color_embed_10000.npy',
                 semantic_centers_path='pretrain/semantic_embed_10000.npy'):
        super().__init__()

        self.encoder = Encoder(encoder_name, ['norm0', 'norm1', 'norm2', 'norm3'], pretrained_path)
        self.encoder.eval()
        test_input = torch.randn(1, num_input_channels, *input_size)
        self.encoder(test_input)
        self.seg_decoder = TRDecoder(
            self.encoder.hooks,
            nf=nf,
            input_channel=num_input_channels,
            ab_segment_classes=ab_segment_classes,
            output_channel=num_output_channels,
            last_norm=last_norm,
            color_centers_path=color_centers_path,
            semantic_centers_path=semantic_centers_path)
        self.refine_net = nn.Sequential(
            *[ResBlock(nf // 2 + 3, norm_type=NormType.Spectral)] * 1,
            custom_conv_layer(nf // 2 + 3, num_output_channels, ks=3, use_activ=False, norm_type=NormType.Spectral))
        self.do_normalize = do_normalize
        self.register_buffer('mean', torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, img):
        return (img - self.mean) / self.std

    def denormalize(self, img):
        return img * self.std + self.mean

    def forward(self, x):
        if x.shape[1] == 3:
            x = self.normalize(x)
        cls = self.encoder(x)
        out_feat = self.seg_decoder(x, cls)

        coarse_input = torch.cat([out_feat, x], dim=1)

        out = self.refine_net(coarse_input)

        if self.do_normalize:
            out = self.denormalize(out)
        return out