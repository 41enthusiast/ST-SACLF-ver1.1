import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import ProjectorBlock, SpatialAttn
from torchvision import models
from collections import OrderedDict


from utils import drop_connect
from pretrained_models import *
from torchvision.models import vgg16
"""
VGG-16 with attention
"""
class AttnVGG(nn.Module): #the vgg n densnet versions
    def __init__(self, num_classes, backbone: VggN, dropout_mode, p, attention=True, normalize_attn=True, model_subtype = 'vgg16'):
        super(AttnVGG, self).__init__()

        self.pretrained = backbone

        self.fhooks = []
        self.selected_out = OrderedDict()
        #self.dense = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=int(sample_size / 32), padding=0,
        #                       bias=True)

        # attention blocks
        self.attention = attention
        if self.attention:
            
            self.project_ins = backbone.project_ins
            for i,p_name in enumerate(['projector0', 'projector1', 'projector2', 'projector3']):
                if backbone.project_ins[i] != 512:
                    setattr(self, p_name, ProjectorBlock(backbone.project_ins[i], 512))

            self.attn0 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)# (batch_size,1,H,W), (batch_size,C)
            self.attn1 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)
            self.attn2 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)
            self.attn3 = SpatialAttn(in_features=512, normalize_attn=normalize_attn)

        #dropout selection for type of regularization
        self.dropout_mode, self.p = dropout_mode, p

        if self.dropout_mode == 'dropout':
            self.dropout = nn.Dropout(self.p)
        elif self.dropout_mode == 'dropconnect':
            self.dropout = drop_connect

        # final classification layer
        if self.attention:
            self.fc1 = nn.Linear(in_features=512*4, out_features=512, bias=True)
            self.classify = nn.Linear(in_features=512, out_features=num_classes, bias=True)
        else:
            self.fc1 = nn.Linear(in_features=512, out_features=256, bias=True)
            self.classify = nn.Linear(in_features=256, out_features=num_classes, bias=True)

    def forward(self, x):
        (l0, l1, l2, l3), g = self.pretrained(x)
        #print(g.shape, l0.shape, l1.shape, l2.shape, l3.shape)
        # attention
        if self.attention:
            for i in range(4):
                if hasattr(self,f'projector{i}'):
                    locals()[f'c{i}'], locals()[f'g{i}'] = getattr(self,f'attn{i}')(getattr(self,f'projector{i}')(locals()[f'l{i}']), g)
                else:
                    locals()[f'c{i}'], locals()[f'g{i}'] = getattr(self,f'attn{i}')(locals()[f'l{i}'], g)
            
            # c0, g0 = self.attn0(self.projector0(l0), g)
            # c1, g1 = self.attn1(self.projector1(l1), g)#this gets it to the same out ch as the next 2 layers
            # c2, g2 = self.attn2(l2, g)
            # c3, g3 = self.attn3(l3, g)
            
            all_locals = locals()
            global_feats = [all_locals[f'g{i}'] for i in range(4)]
            attn_maps = [all_locals[f'c{i}'] for i in range(4)]
            g = torch.cat(global_feats, dim=1) # batch_sizex3C

            # fc layer
            out = torch.relu(self.fc1(g)) # batch_sizexnum_classes

            if self.dropout_mode == 'dropout':
                out = self.dropout(out)
            elif self.dropout_mode == 'dropconnect':
                out = self.dropout(out, self.p, self.training)

        else:
            attn_maps = [None, None, None, None]
            out = self.fc1(torch.squeeze(g))

            if self.dropout_mode == 'dropout':
                out = self.dropout(out)
            elif self.dropout_mode == 'dropconnect':
                out = self.dropout(out, self.p, self.training)

        out = self.classify(out)
        return [out,]+ attn_maps



class AttnResNet(nn.Module): #the vgg n densnet versions
    def __init__(self, num_classes, backbone:ResNetN, dropout_mode, p, attention=True, normalize_attn=True):
        super(AttnResNet, self).__init__()
        # conv blocks
        self.pretrained = backbone

        self.fhooks = []
        self.selected_out = OrderedDict()

        # attention blocks
        self.attention = attention
        if self.attention:

            self.project_ins = backbone.project_ins
            for i,p_name in enumerate(['projector0', 'projector1', 'projector2', 'projector3']):
                if backbone.project_ins[i] != backbone.project_ins[-1]:
                    setattr(self, p_name, ProjectorBlock(backbone.project_ins[i], backbone.project_ins[-1]))

            self.attn0 = SpatialAttn(in_features=backbone.project_ins[-1], normalize_attn=normalize_attn)# (batch_size,1,H,W), (batch_size,C)
            self.attn1 = SpatialAttn(in_features=backbone.project_ins[-1], normalize_attn=normalize_attn)
            self.attn2 = SpatialAttn(in_features=backbone.project_ins[-1], normalize_attn=normalize_attn)
            self.attn3 = SpatialAttn(in_features=backbone.project_ins[-1], normalize_attn=normalize_attn)

        # dropout selection for type of regularization
        self.dropout_mode, self.p = dropout_mode, p

        if self.dropout_mode == 'dropout':
            self.dropout = nn.Dropout(self.p)
        elif self.dropout_mode == 'dropconnect':
            self.dropout = drop_connect

        # final classification layer
        if self.attention:
            self.fc1 = nn.Linear(in_features=backbone.project_ins[-1] * 4, out_features=backbone.project_ins[-1], bias=True)
            self.classify = nn.Linear(in_features=backbone.project_ins[-1], out_features=num_classes, bias=True)
        else:
            self.fc1 = nn.Linear(in_features=backbone.project_ins[-1], out_features=backbone.project_ins[-1]//2, bias=True)
            self.classify = nn.Linear(in_features=backbone.project_ins[-1]//2, out_features=num_classes, bias=True)


    def forward_hook(self, layer_name):
        def hook(module, input, output):
            self.selected_out[layer_name] = output
        return hook

    def forward(self, x):

        l0, l1, l2, l3, g = self.pretrained(x)

        # attention
        if self.attention:
            for i in range(4):
                if hasattr(self,f'projector{i}'):
                    locals()[f'c{i}'], locals()[f'g{i}'] = getattr(self,f'attn{i}')(getattr(self,f'projector{i}')(locals()[f'l{i}']), g)
                else:
                    locals()[f'c{i}'], locals()[f'g{i}'] = getattr(self,f'attn{i}')(locals()[f'l{i}'], g)
            
            all_locals = locals()
            global_feats = [all_locals[f'g{i}'] for i in range(4)]
            attn_maps = [all_locals[f'c{i}'] for i in range(4)]
            g = torch.cat(global_feats, dim=1) # batch_sizex3C

            # fc layer
            out = torch.relu(self.fc1(g)) # batch_sizexnum_classes

            if self.dropout_mode == 'dropout':
                out = self.dropout(out)
            elif self.dropout_mode == 'dropconnect':
                out = self.dropout(out, self.p, self.training)

        else:
            attn_maps = [None, None, None, None]
            out = self.fc1(torch.squeeze(g))

            if self.dropout_mode == 'dropout':
                out = self.dropout(out)
            elif self.dropout_mode == 'dropconnect':
                out = self.dropout(out, self.p, self.training)

        out = self.classify(out)
        return [out,]+ attn_maps

        
#MULTITASK MODELS - model constructs itself from the task flows




# Test
if __name__ == '__main__':
    Vgg16([2, 9, 22, 30])
    model = AttnVGG(num_classes=10,backbone=vgg16(pretrained=True).features, dropout_mode='dropout', p=0.2, model_subtype='vgg13')
    x = torch.randn(16,3,256,256)
    opt = torch.optim.Adam(model.parameters())

    out, c0, c1, c2, c3 = model(x)
    print('VGG', out.shape, c0.shape, c1.shape, c2.shape, c3.shape)

    #model = AttnVGG(num_classes=10, output_layers=[0, 7, 21, 28], dropout_mode='dropconnect', p=0.2)
    #x = torch.randn(16,3,256,256)
    #out, c0, c1, c2, c3 = model(x)
    #print('VGG', out.shape, c0.shape, c1.shape, c2.shape, c3.shape)

    #model = AttnResnet(num_classes=10, output_layers=['0', '4.1.4', '6.2.2', '7.1.2'], dropout_mode='dropout', p=0.2)
    #x = torch.randn(16, 3, 256, 256)
    #out, c0, c1, c2, c3 = model(x)
    #print('Resnet', out.shape, c0.shape, c1.shape, c2.shape, c3.shape)