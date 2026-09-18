"""Independently composed decoder using TorchVision's licensed ConvNeXt-Tiny.
Existing architecture family, not a proposed original model.
"""
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import convnext_tiny
from pretrained_unet import Block,plain,setup_pretrained_data,make_optimizer,ImageNetNormalize
ROOT=Path(__file__).resolve().parents[2]
WEIGHT_SHA='983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d'

class PretrainedUNet(nn.Module):
    def __init__(self):
        super().__init__();path=ROOT/'data/pretrained_weights/convnext_tiny-983f1562.pth'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==WEIGHT_SHA
        self.encoder=convnext_tiny(weights=None,stochastic_depth_prob=.1)
        self.encoder.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        self.encoder.classifier=nn.Identity()
        self.shallow=Block(3,16);self.half_res=Block(3,32)
        self.decoders=nn.ModuleList([Block(1152,256),Block(448,128),Block(224,64),Block(96,32),Block(48,32)])
        self.output=nn.Conv2d(32,2,1)
    def forward(self,image):
        x=plain(image);z=x;stages=[]
        for index,layer in enumerate(self.encoder.features):
            z=layer(z)
            if index in (1,3,5,7):stages.append(z)
        a,b,c,z=stages
        skip_half=self.half_res(F.avg_pool2d(x,2))
        for block,skip in zip(self.decoders,[c,b,a,skip_half,self.shallow(x)]):
            z=block(torch.cat((F.interpolate(z,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip),1))
        return self.output(z)
