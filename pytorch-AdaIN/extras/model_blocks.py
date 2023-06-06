import torch
from .utils import *

# #LAYERS
#
# class ConvLayer(torch.nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride):
#         super(ConvLayer, self).__init__()
#         reflection_padding = kernel_size // 2  # same dimension after padding
#         self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
#         self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)  # remember this dimension
#
#     def forward(self, x):
#         out = self.reflection_pad(x)
#         out = self.conv2d(out)
#         return out
#
# class UpsampleConvLayer(torch.nn.Module):
#     """UpsampleConvLayer
#     Upsamples the input and then does a convolution. This method gives better results
#     compared to ConvTranspose2d.
#     ref: http://distill.pub/2016/deconv-checkerboard/
#     """
#
#     def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
#         super(UpsampleConvLayer, self).__init__()
#         self.upsample = upsample
#         if upsample:
#             self.upsample_layer = torch.nn.Upsample(mode='nearest', scale_factor=upsample)
#         reflection_padding = kernel_size // 2
#         self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
#         self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)
#
#     def forward(self, x):
#         x_in = x
#         if self.upsample:
#             x_in = self.upsample_layer(x_in)
#         out = self.reflection_pad(x_in)
#         out = self.conv2d(out)
#         return out
#
# #BLOCKS
#
# class ResidualBlock(torch.nn.Module):
#     """ResidualBlock
#     introduced in: https://arxiv.org/abs/1512.03385
#     recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
#     """
#
#     def __init__(self, channels):
#         super(ResidualBlock, self).__init__()
#         self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
#         self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
#         self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
#         self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
#         self.relu = torch.nn.ReLU()
#
#     def forward(self, x):
#         residual = x
#         out = self.relu(self.in1(self.conv1(x)))
#         out = self.in2(self.conv2(out))
#         out = out + residual  # need relu right after
#         return out

#
# class AdaptiveInstanceNorm2d(torch.nn.Module):
#     """
#     Adaptive Instance Normalization
#     """
#
#     def __init__(self, style_num, in_channels):
#         super(AdaptiveInstanceNorm2d, self).__init__()
#         self.inns = torch.nn.ModuleList([torch.nn.InstanceNorm2d(in_channels, affine=True) for i in range(style_num)])#the magic for multiple styles
#
#     def forward(self, x, style_id, y):
#         out = torch.stack([y[i].std()*self.inns[style_id[i]](x[i].unsqueeze(0)).squeeze_(0) +y[i].mean() for i in range(len(style_id))])#multiple styles
#         return out

#
# class AdaIN:
#     """
#     Adaptive Instance Normalization as proposed in
#     'Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization' by Xun Huang, Serge Belongie.
#     """
#
#     def _compute_mean_std(
#         self, feats: torch.Tensor, eps=1e-8, infer=False
#     ) -> torch.Tensor:
#         return calc_mean_std(feats, eps)
#
#     def __call__(
#         self,
#         c_feats: torch.Tensor,
#         s_feats: torch.Tensor,
#         infer: bool = False,
#     ) -> torch.Tensor:
#         """
#         __call__ Adaptive Instance Normalization as proposaed in
#         'Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization' by Xun Huang, Serge Belongie.
#         Args:
#             content_feats (torch.Tensor): Content features
#             style_feats (torch.Tensor): Style Features
#         Returns:
#             torch.Tensor: [description]
#         """
#         sz = c_feats.shape
#         c_mean, c_std = self._compute_mean_std(c_feats, infer=infer)
#         s_mean, s_std = self._compute_mean_std(s_feats, infer=infer)
#         if torch.isnan(c_std).any() or torch.isnan(s_std).any():
#             print('Naan',c_feats.shape, s_feats.shape)
#             quit(-1)
#         #print(c_feats.shape, s_std.shape, c_std.shape, c_mean.shape, s_mean.shape)
#         normalized = (s_std.expand(sz) *
#                       (c_feats - c_mean.expand(sz))
#                       / c_std.expand(sz)) \
#                      + s_mean.expand(sz)
#
#         return normalized