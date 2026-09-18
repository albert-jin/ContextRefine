"""Shared head correction across controls/candidates, not a method contribution."""
from torch import nn
from monai.networks.nets import UNet
class FeatureUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=UNet(spatial_dims=2,in_channels=3,out_channels=16,channels=(16,32,64,128,256),
                           strides=(2,2,2,2),num_res_units=2,norm='INSTANCE')
        self.classifier=nn.Conv2d(16,2,1)
    def forward(self,x):return self.classifier(self.features(x))
