import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image

import net
from function import adaptive_instance_normalization, coral

from torch.utils.data import Subset, DataLoader
from torchvision.datasets import ImageFolder
from sampler import InfiniteSamplerWrapper

def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


def style_transfer(vgg, decoder, content_f, style_f, alpha=1.0,
                   interpolation_weights=None):
    assert (0.0 <= alpha <= 1.0)
    # content_f = vgg(content)
    # style_f = vgg(style)
    if interpolation_weights:
        _, C, H, W = content_f.size()
        feat = torch.FloatTensor(1, C, H, W).zero_().to(device)
        base_feat = adaptive_instance_normalization(content_f, style_f)
        for i, w in enumerate(interpolation_weights):
            feat = feat + w * base_feat[i:i + 1]
        content_f = content_f[0:1]
    else:
        feat = adaptive_instance_normalization(content_f[-1], style_f[-1])
        #feat = adaptive_instance_normalization(content_f, style_f)
    feat = feat * alpha + content_f[-1] * (1 - alpha)
    #feat = feat * alpha + content_f * (1 - alpha)
    return decoder(feat)


parser = argparse.ArgumentParser()
# Basic options
# parser.add_argument('--content', type=str,
#                     help='File path to the content image')
# parser.add_argument('--content_dir', type=str,
#                     help='Directory path to a batch of content images')
# parser.add_argument('--style', type=str,
#                     help='File path to the style image, or multiple style \
#                     images separated by commas if you want to do style \
#                     interpolation or spatial control')
# parser.add_argument('--style_dir', type=str,
#                     help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='models/vgg_normalised.pth')
parser.add_argument('--decoder', type=str, default='models/decoder.pth')

# Additional options
parser.add_argument('--content_size', type=int, default=512,
                    help='New (minimum) size for the content image, \
                    keeping the original size if set to 0')
parser.add_argument('--style_size', type=int, default=512,
                    help='New (minimum) size for the style image, \
                    keeping the original size if set to 0')
parser.add_argument('--crop', action='store_true',
                    help='do center crop to create squared image')
parser.add_argument('--save_ext', default='.jpg',
                    help='The extension name of the output image')
parser.add_argument('--output', type=str, default='output',
                    help='Directory to save the output image(s)')

# Advanced options
parser.add_argument('--preserve_color', action='store_true',
                    help='If specified, preserve color of the content image')
parser.add_argument('--alpha', type=float, default=1.0,
                    help='The weight that controls the degree of \
                             stylization. Should be between 0 and 1')
parser.add_argument(
    '--style_interpolation_weights', type=str, default='',
    help='The weight for blending the style of multiple style images')

args = parser.parse_args()

do_interpolation = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



decoder = net.decoder
vgg = net.vgg

decoder.eval()
vgg.eval()

decoder.load_state_dict(torch.load(args.decoder))

vgg.to(device)
decoder.to(device)

content_tf = test_transform(args.content_size, args.crop)

dataset_root = '../kaokore_imagenet_style/status/train'
                
batch_size, n_threads = 8, 16


train_dataset = ImageFolder(dataset_root, transform=transforms.ToTensor())
for i, tgt in enumerate('commoner  incarnation  noble  warrior'.split('  ')):
    indices = list(
        filter(lambda idx: train_dataset[idx][1] == i,
               range(len(train_dataset)))
    )
    content_dataset = Subset(train_dataset, indices)
    print(len(content_dataset))
    content_iter = DataLoader(
        content_dataset, batch_size=batch_size,
        num_workers=n_threads)
    style_iter = DataLoader(
        content_dataset, batch_size=batch_size,
        shuffle=True, num_workers=n_threads)

    output_dir = Path(f'../kaokore_control_v1/{tgt}')
    output_dir.mkdir(exist_ok=True, parents=True)

    for i in range(len(content_dataset)):
        content = content_dataset[torch.randint(len(content_dataset), (1,))][0]
        style = content_dataset[torch.randint(len(content_dataset), (1,))][0]
        if args.preserve_color:
            for j in range(len(style)):
                style[j] = coral(style[j], content[j])
        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)
        with torch.no_grad():
            content_f, _ = vgg(content)
            style_f, _ = vgg(style)
            output = style_transfer(vgg, decoder, content_f, style_f,
                                    args.alpha)
        output = output.cpu()
        for j, o_img in enumerate(output):
            output_name = output_dir / f'stylized_{i * batch_size + j}{args.save_ext}'
            save_image(o_img, str(output_name))

