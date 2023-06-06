from PIL import Image
import os
import torch
import torch.nn.functional as F
import sys
from torch import nn

def check_paths(args):
    try:
        if not os.path.exists(args.save_model_dir):
            os.makedirs(args.save_model_dir)
        if args.checkpoint_model_dir is not None and not (os.path.exists(args.checkpoint_model_dir)):
            os.makedirs(args.checkpoint_model_dir)
    except OSError as e:
        print(e)
        sys.exit(1)

def load_image(filename, size=None, scale=None):
    img = Image.open(filename)
    if size is not None:
        img = img.resize((size, size), Image.ANTIALIAS)
    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    return img


def save_img(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1) # new_tensor for same dimension of tensor
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0) # back to tensor within 0, 1
    return (batch - mean) / std


#drops images in a batch from being seen by a subset of layers
def drop_connect(inputs, p, train_mode):  # bchw, 0-1, bool
    assert 0 <= p <= 1, 'p must be in range of [0,1]'

    #inference
    if not train_mode:
        return inputs

    bsz = inputs.shape[0]

    keep_prob = 1-p

    #binary mask for selection of weights
    rand_tensor = keep_prob
    rand_tensor += torch.rand([bsz, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    mask = torch.floor(rand_tensor)

    outputs = inputs / keep_prob*mask
    return outputs


def focal_loss(n_classes, gamma=2., alpha=4.):
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y, y_pred):
        eps = 1e-9
        pred = torch.softmax(y_pred, dim=1) + eps
        #pred = y_pred + eps
        y_true = F.one_hot(y, n_classes)
        cross_entropy = y_true * -1*torch.log(pred)
        wt = y_true*(1-pred)**gamma
        focal_loss = alpha*wt*cross_entropy
        focal_loss = torch.max(focal_loss, dim=1)[0]
        return focal_loss.mean()
    return focal_loss_fixed

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2) # swapped ch and w*h, transpose share storage with original
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

# def compute_mean_std(
#     feats: torch.Tensor, eps=1e-8, infer=False
# ) -> torch.Tensor:
#     assert (
#         len(feats.shape) == 4
#     ), "feature map should be 4-dimensional of the form N,C,H,W!"
#     #  * Doing this to support ONNX.js inference.
#     if infer:
#         n = 1
#         c = 512  # * fixed for vgg19
#     else:
#         n, c, _, _ = feats.shape
#     feats = feats.view(n, c, -1)
#     mean = torch.mean(feats, dim=-1).view(n, c, 1, 1)
#     std = torch.std(feats, dim=-1, unbiased=True).view(n, c, 1, 1) + eps
#     # if torch.isnan(std).any():
#     #     print(feats.shape[2])
#     #     std = feats.view(n,c,1,1)
#     return mean, std

def adjust_learning_rate(optimizer, iteration_count, lr, lr_decay=5e-5):
    """Imitating the original implementation"""
    lr = lr / (1.0 + lr_decay * iteration_count)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)

mse_loss = nn.MSELoss()

def calc_content_loss( input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    return mse_loss(input, target)

def calc_style_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mse_loss(input_mean, target_mean) + \
           mse_loss(input_std, target_std)