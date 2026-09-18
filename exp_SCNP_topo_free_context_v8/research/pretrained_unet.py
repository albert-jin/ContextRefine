"""Known ResNet34+U-Net baseline, independently composed, not a novel module."""
import hashlib,json
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet34
from monai.data import Dataset
from monai.transforms import Compose,NormalizeIntensityd,ScaleIntensityRanged

ROOT=Path(__file__).resolve().parents[2]
WEIGHT_SHA='b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f'
def plain(x):return x.as_tensor() if hasattr(x,'as_tensor') else torch.as_tensor(x)

class ImageNetNormalize:
    def __init__(self,from_bytes=False):self.from_bytes=from_bytes
    def __call__(self,item):
        row=dict(item);x=plain(row['image']).float()
        if self.from_bytes:x=x/255.
        x=x.clamp(0,1)
        row['image']=(x-x.new_tensor([.485,.456,.406])[:,None,None])/x.new_tensor([.229,.224,.225])[:,None,None]
        return row

def setup_pretrained_data(seed):
    from train_strong import setup_data
    tr,va,manifest,cfg=setup_data(seed)
    transforms=list(tr.transform.transforms)
    assert isinstance(transforms[0],NormalizeIntensityd)
    transforms[0]=ScaleIntensityRanged(['image'],a_min=0,a_max=255,b_min=0,b_max=1,clip=True)
    transforms.append(ImageNetNormalize())
    transform=Compose(transforms);transform.set_random_state(seed=seed)
    return Dataset(tr.data,transform),Dataset(va.data,ImageNetNormalize(from_bytes=True)),manifest,cfg

class Block(nn.Sequential):
    def __init__(self,cin,cout):
        super().__init__(nn.Conv2d(cin,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.GELU(),
                         nn.Conv2d(cout,cout,3,padding=1,bias=False),nn.GroupNorm(8,cout),nn.GELU())

class PretrainedUNet(nn.Module):
    def __init__(self):
        super().__init__();path=ROOT/'data/pretrained_weights/resnet34-b627a593.pth'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==WEIGHT_SHA
        self.encoder=resnet34(weights=None)
        self.encoder.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
        self.encoder.fc=nn.Identity()
        self.shallow=Block(3,16)
        self.decoders=nn.ModuleList([Block(768,256),Block(384,128),Block(192,64),Block(128,32),Block(48,32)])
        self.output=nn.Conv2d(32,2,1)
    def forward(self,image):
        x=plain(image);e=self.encoder
        stem=e.relu(e.bn1(e.conv1(x)));a=e.layer1(e.maxpool(stem));b=e.layer2(a);c=e.layer3(b);z=e.layer4(c)
        for block,skip in zip(self.decoders,[c,b,a,stem,self.shallow(x)]):
            z=block(torch.cat((F.interpolate(z,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip),1))
        return self.output(z)

def make_optimizer(model):
    decoder=[p for name,p in model.named_parameters() if not name.startswith('encoder.')]
    return torch.optim.AdamW([{'params':model.encoder.parameters(),'lr':3e-5},{'params':decoder,'lr':3e-4}],weight_decay=.01)
