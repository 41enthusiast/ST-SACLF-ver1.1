import numpy as np
import torch
from torchvision.models import vgg16, vgg19, vgg13
from torchvision.models import VGG16_Weights, VGG19_Weights, VGG13_Weights

import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageNet

output_filename = 'models/vgg13_normalized.pth'
batch_size = 16
img_size = (256,256)

def layer_mean_activations(x):
    return x.mean(dim = (0,2,3))

if __name__ == '__main__':
    #model = vgg16(weights = VGG16_Weights.DEFAULT).features
    #model = vgg19(weights=VGG19_Weights.DEFAULT).features
    model = vgg13(weights=VGG13_Weights.DEFAULT).features.cuda()

    normalize = transforms.Compose( [transforms.ToTensor(),
                                    transforms.Normalize(mean = [0.485, 0.456, 0.406],
                                                         std = [0.229, 0.224, 0.225]
                                                         ),
                                     transforms.Resize(img_size)]
                                    )
    imagenet_ds = ImageNet(root = 'data/imagenet', split='val', transform=normalize)
    imagenet_loader = DataLoader(imagenet_ds, batch_size = batch_size)
    print('Loaded dataset')

    #get intermediate weights
    hooks = {}
    def get_activations(layer_name):
        def hook(module, input, output):
            global layer_activations
            layer_activations[layer_name] = output
        return hook
    layer_activations = {}
    for name, module in model.named_modules():
        hooks[name] = module.register_forward_hook(get_activations(name))

    print('Calculating the mean activations over the imagenet val activations')
    with torch.no_grad():
        mean_activations = []
        for nbatch, (x, _) in enumerate(imagenet_loader, 1):
            model(x.cuda())
            for i, (module_name, lyr) in enumerate(model.named_modules()):
                if nbatch==1:
                    mean_activations.append(layer_mean_activations(layer_activations[module_name]))
                else:
                    mean_activations[i] += layer_mean_activations(layer_activations[module_name])
    mean_activations = iter([m_act/nbatch for m_act in mean_activations])

    print('Normalizing the mean activations')
    prev_conv_lyr_means = None
    adapted_state_space = {}
    for i, (module_name, lyr) in enumerate(model.named_parameters()):
        adapted_state_space[module_name] = lyr
        if 'Conv' in module_name:
            means = next(mean_activations)
            if prev_conv_lyr_means is not None:
                if 'weight' in module_name:
                    adapted_state_space[module_name]*=prev_conv_lyr_means.expand(0,2,3)
            adapted_state_space[module_name]/= means
            prev_conv_lyr_means = means

    print('Saving model weights')
    model.load_state_dict(adapted_state_space)
    torch.save(adapted_state_space, output_filename)