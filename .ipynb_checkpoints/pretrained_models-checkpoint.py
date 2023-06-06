import torch
from torch import nn
from torchvision import models
from collections import namedtuple
from collections import OrderedDict
from torchvision.models.resnet import BasicBlock, Bottleneck, ResNet
from typing import Type, Any, Union, List

from torch import nn
import torch.nn.functional as F

from collections import namedtuple


from torchvision import models
#from torchvision.models import VGG16_Weights
from model_blocks import *

import torch

def get_project_in(model: torch.nn.Sequential):
    out_dim = 0
    for layer in model.children():
        if 'Conv2d' in str(type(layer)):
            out_dim = layer.out_channels
        elif 'BasicBlock' in str(type(layer)):
            out_dim = get_project_in(layer)
        elif 'Bottleneck' in str(type(layer)):
            out_dim = get_project_in(layer)
        elif 'Sequential' in str(type(layer)):
            out_dim = get_project_in(layer)
    return out_dim

#the pretrained model
class Vgg(torch.nn.Module):
    def __init__(self,  output_layers, requires_grad=False):
        super(Vgg, self).__init__()
        vgg = models.vgg19().features
        vgg_pretrained_features = vgg

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()

        reflect = nn.ReflectionPad2d((1, 1, 1, 1))

        dense = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.slice5 = torch.nn.Sequential()#global features
        # old_pretrained_state_space = torch.load('models/vgg_normalised.pth')
        # temp_state_space = {'0.weight': old_pretrained_state_space['0.weight'], '0.bias': old_pretrained_state_space['0.bias']}
        extra = torch.nn.Conv2d(3, 3, (1, 1))
        self.slice1.add_module(str(0), extra)
        self.slice1.add_module(str(1), reflect)
        # self.slice1.load_state_dict(temp_state_space)
        for x in range(2, output_layers[0]+1+2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x-2])
        for x in range(output_layers[0]+1, output_layers[1]+1):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(output_layers[1]+1, output_layers[2]+1):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(output_layers[2]+1, output_layers[3]+1):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(output_layers[3]+1, len(vgg_pretrained_features)-1):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        self.slice5.add_module(str(len(vgg_pretrained_features)-1), dense)

        pretrained_state_space = torch.load('models/vgg_normalised.pth')
        temp_state_space = self.state_dict()
        for k_p, k_n in zip(list(pretrained_state_space.keys()), list(temp_state_space.keys())):
            temp_state_space[k_n] = pretrained_state_space[k_p]
        self.load_state_dict(temp_state_space)

        self.project_ins = [get_project_in(self.slice1),]
        self.project_ins.append(get_project_in(self.slice2))
        self.project_ins.append(get_project_in(self.slice3))
        self.project_ins.append(get_project_in(self.slice4))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        l0 = h
        h = self.slice2(h)
        l1 = h
        h = self.slice3(h)
        l2 = h
        h = self.slice4(h)
        l3 = h

        g = self.slice5(h)

        vgg_outputs = namedtuple("VggOutputs", ['style_lyr1', 'style_lyr2', 'style_lyr3', 'content_lyr'])
        out = vgg_outputs(l0, l1, l2, l3)

        return out, g

class VggN(torch.nn.Module):
    def __init__(self, output_layers, model_type = 'vgg16',requires_grad=False):
        super(VggN, self).__init__()
        if model_type == 'vgg16':
            vgg = models.vgg16(pretrained=True)
        elif model_type == 'vgg19':
            vgg = models.vgg19(pretrained=True)
        elif model_type == 'vgg13':
            vgg = models.vgg13(pretrained=True)
        vgg_pretrained_features = vgg.features

        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        dense = torch.nn.AdaptiveAvgPool2d((1, 1))

        for x in range(0, output_layers[0]+1):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(output_layers[0]+1, output_layers[1]+1):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(output_layers[1]+1, output_layers[2]+1):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(output_layers[2]+1, output_layers[3]+1):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(output_layers[3]+1, len(vgg_pretrained_features)-1):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        self.slice5.add_module(str(len(vgg_pretrained_features)-1), dense)

        self.project_ins = [get_project_in(self.slice1),]
        self.project_ins.append(get_project_in(self.slice2))
        self.project_ins.append(get_project_in(self.slice3))
        self.project_ins.append(get_project_in(self.slice4))

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        l0 = h
        h = self.slice2(h)
        l1 = h
        h = self.slice3(h)
        l2 = h
        h = self.slice4(h)
        l3 = h

        g = self.slice5(h)

        vgg_outputs = namedtuple("VggOutputs", ['style_lyr1', 'style_lyr2', 'style_lyr3', 'content_lyr'])
        out = vgg_outputs(l0, l1, l2, l3)

        return out, g


# class TransformerDecoder(nn.Module):
#


decoder = nn.Sequential(
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),
    # nn.ReflectionPad2d((1, 1, 1, 1)),
    # nn.Conv2d(512, 512, (3, 3)),
    # nn.ReLU(),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),

    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

class Vgg16Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        block1 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )

        block2 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )

        block3 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(256, 128, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )
        block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
        )
        block5 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, padding_mode="reflect"),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, 1, 1, padding_mode="reflect"),
            #nn.ReLU(),
        )

        self.net = nn.ModuleList([block1, block2, block3, block4, block5])

    def forward(self, x):
        for ix, module in enumerate(self.net):
            x = module(x)
            #print(x.shape)
            if ix < len(self.net) -1:#at max pool locations
                x = F.interpolate(x, scale_factor= 2, mode='nearest')
        return x
#VGG-13


#VGG-19


#ResNet-N
class IntResNet(ResNet):
    def __init__(self,output_layer, output_layers, *args):
        self.output_layer = output_layer
        super().__init__(*args)
        
        self._layers = []
        self.project_in = []
        for l in list(self._modules.keys()):
            self._layers.append(l)
            if l == output_layer:
                break
        self.layers = OrderedDict(zip(self._layers,[getattr(self,l) for l in self._layers]))

        for lyr in self.layers:
          if lyr in output_layers:
            if 'Conv2d' in str(type(self.layers[lyr])):
              self.project_in.append(self.layers[lyr].out_channels)   
            elif 'Sequential' in str(type(self.layers[lyr])):
              self.project_in.append(get_project_in(self.layers[lyr]))

        self.output_layers = output_layers

    def _forward_impl(self, x):
        outputs = []
        for l in self._layers:
            x = self.layers[l](x)
            if l in self.output_layers:
              outputs.append(x)

        return outputs, x

    def forward(self, x):
        return self._forward_impl(x)

class ResNetN(nn.Module):
    # base_model : The model we want to get the output from
    # base_out_layer : The layer we want to get output from
    # num_trainable_layer : Number of layers we want to finetune (counted from the top)
    #                       if enetered value is -1, then all the layers are fine-tuned
    def __init__(self,base_model,base_out_layer, base_out_lyrs, requires_grad = False):
        super().__init__()
        self.base_model = base_model
        self.base_out_layer = base_out_layer
        self.base_out_lyrs = base_out_lyrs
        
        self.model_dict = {
                           'resnet34':{'block':BasicBlock,'layers':[3,4,6,3],'kwargs':{}},
                           'resnet50':{'block':Bottleneck,'layers':[3,4,6,3],'kwargs':{}},
                           'resnet101':{'block':Bottleneck,'layers':[3,4,23,3],'kwargs':{}},
                           }
        
        #PRETRAINED MODEL
        self.resnet = self.new_resnet(self.base_model,self.base_out_layer,self.base_out_lyrs,
                                     self.model_dict[self.base_model]['block'],
                                     self.model_dict[self.base_model]['layers'],
                                     True,True,
                                     **self.model_dict[self.base_model]['kwargs'])

        self.layers = list(self.resnet._modules.keys())#has the truncated model
        print(self.layers)
        self.project_ins = self.resnet.project_in
        #FREEZING LAYERS
        self.total_children = 0
        self.children_counter = 0
        for c in self.resnet.children():
            self.total_children += 1
        
        for c in self.resnet.children():
            for param in c.parameters():
                param.requires_grad = requires_grad
            self.children_counter += 1
                    
    def new_resnet(self,
                   model_type: str,
                   outlayer: str,
                   outLayers: List[str],
                   block: Type[Union[BasicBlock, Bottleneck]],
                   layers: List[int],
                   pretrained: bool,
                   progress: bool,
                   **kwargs: Any
                  ) -> IntResNet:

        '''model_urls = {
            'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
            'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
            'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
        }'''
        model = IntResNet(outlayer, outLayers, block, layers, **kwargs)
        model_keys = model.layers.keys()
        if pretrained:
            # state_dict = load_state_dict_from_url(model_urls[arch],
            #                                       progress=progress)
            if model_type == 'resnet34':
                resnet = models.resnet34(pretrained=True)
            elif model_type == 'resnet50':
                resnet = models.resnet50(pretrained=True)
            elif model_type == 'resnet101':
                resnet = models.resnet101(pretrained=True)
            model.load_state_dict(resnet.state_dict())
            for k in model._modules.keys():
              if k not in model_keys:
                del model._modules[k] 
        return model
    
    def forward(self,x):
        (l0, l1, l2, l3), g = self.resnet(x)
        return l0, l1, l2, l3 ,g

if __name__ == '__main__':
    temp_img = torch.randn((8, 3, 255, 255))
    vgg_output_lyrs = {'og':[2, 9, 22, 30],
                       'st':[1, 8, 13, 20],
                       'st-vgg13':[1, 6, 11, 16],
                       'vgg13': [1, 8, 18, 23]}['og']
    temp = VggN(vgg_output_lyrs)
    print(temp)
    #print(temp(temp_img)[1].shape)
