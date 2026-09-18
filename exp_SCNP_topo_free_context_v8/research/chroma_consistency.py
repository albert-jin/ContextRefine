"""Training-only paired chromatic views and structure-weighted consistency.

Consistency under augmentation is established, e.g. AugMix (ICLR2020):
https://arxiv.org/abs/1912.02781 . This is an independent two-view adaptation,
not an exact AugMix reproduction or a claim of inventing consistency losses.
"""
import torch
from torch import nn

WARM_REL='exp_SCNP_topo_visibility_budget_v6/auto_res_logs/runs/topo_visibility_pair_base_s0/best.pt'
WARM_SHA='eed07a6e1185c59e6f2cc5e9591641fa5a708089806a58fdfc2feb8bce22ec6e'

def plain(x):return x.as_tensor() if hasattr(x,'as_tensor') else x

def colors(x):
    x=plain(x).float()
    mu=x.new_tensor([.485,.456,.406])[None,:,None,None]
    sd=x.new_tensor([.229,.224,.225])[None,:,None,None]
    return (x*sd+mu).clamp(0,1),mu,sd

@torch.no_grad()
def paired_view(x,generator):
    rgb,mu,sd=colors(x);n=len(rgb)
    def rand(channels=1):return torch.rand(n,channels,1,1,device=rgb.device,generator=generator)
    # Every draw uses a dedicated stream shared across experimental arms.
    gamma=.65+.85*rand();gain=.6+.8*rand(3)
    z=rgb.clamp_min(1e-6).pow(gamma)*gain
    gray=(z*z.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    saturation=1.5*rand();saturation=torch.where(rand()<.35,torch.zeros_like(saturation),saturation)
    z=gray+saturation*(z-gray)
    permutation=torch.rand(n,3,device=z.device,generator=generator).argsort(1)
    permuted=z.gather(1,permutation[:,:,None,None].expand_as(z))
    z=torch.where(rand()<.25,permuted,z)
    return (z.clamp(0,1)-mu)/sd

def structure_js(z1,z2,labels,geometry):
    """Symmetric JS, normalized within each present class and image.

    GT geometry weights lie in [1,5]. Empty classes contribute no denominator
    or loss. No cross-image normalization, pseudo labels or test data are used.
    """
    p=plain(z1).float().softmax(1).clamp_min(1e-7)
    q=plain(z2).float().softmax(1).clamp_min(1e-7)
    m=.5*(p+q)
    js=(.5*(p*(p.log()-m.log())+q*(q.log()-m.log()))).sum(1,keepdim=True).clamp_min(0)
    y=plain(labels).float();w=(1+4*plain(geometry).float().clamp(0,1)).detach()
    values=[];present=[]
    for mask in (y,1-y):
        wm=w*mask;denom=wm.sum((1,2,3));ok=denom>0
        values.append((js*wm).sum((1,2,3))/denom.clamp_min(1e-12))
        present.append(ok.float())
    return ((values[0]+values[1])/(present[0]+present[1]).clamp_min(1)).mean()

def freeze_encoder_bn(model):
    for module in model.encoder.modules():
        if isinstance(module,nn.modules.batchnorm._BatchNorm):module.eval()

def stress_views(x):
    if hasattr(x,'as_tensor'):x=x.as_tensor()
    mu=x.new_tensor([.485,.456,.406])[None,:,None,None]
    sd=x.new_tensor([.229,.224,.225])[None,:,None,None]
    rgb=(x*sd+mu).clamp(0,1)
    gray=(rgb*x.new_tensor([.299,.587,.114])[None,:,None,None]).sum(1,keepdim=True).expand_as(rgb)
    cycle=rgb[:,[1,2,0]]
    gradient=torch.linspace(.55,1.,rgb.shape[-1],device=x.device,dtype=x.dtype)[None,None,None,:]
    warm=(rgb*x.new_tensor([1.12,.92,.78])[None,:,None,None]*gradient).clamp(0,1)
    return {k:(v-mu)/sd for k,v in {'grayscale':gray,'rgb_cycle':cycle,'warm_shadow':warm}.items()}

def self_test():
    torch.manual_seed(4)
    a=torch.randn(3,2,9,11,requires_grad=True);b=torch.randn_like(a,requires_grad=True)
    y=torch.randint(0,2,(3,1,9,11)).float();geometry=torch.rand_like(y)
    identical=structure_js(a,a,y,geometry);assert float(identical.detach())<1e-7
    forward=structure_js(a,b,y,geometry);reverse=structure_js(b,a,y,geometry)
    torch.testing.assert_close(forward,reverse);assert 0<float(forward.detach())<.6932
    forward.backward();assert all(x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum()>0 for x in (a,b))
    for yy in (torch.zeros_like(y),torch.ones_like(y)):
        assert torch.isfinite(structure_js(a,b,yy,geometry))
    x=torch.randn(4,3,13,17)
    v1=paired_view(x,torch.Generator().manual_seed(12));v2=paired_view(x,torch.Generator().manual_seed(12))
    assert v1.shape==x.shape and torch.equal(v1,v2) and torch.isfinite(v1).all()
    c=paired_view(torch.ones(2,3,9,11),torch.Generator().manual_seed(1))
    assert torch.equal(c[:,:,0:1,0:1].expand_as(c),c),'Color augmentation changed spatially constant input'
    assert all(v.shape==x.shape and torch.isfinite(v).all() for v in stress_views(x).values())
    class Model(nn.Module):
        def __init__(self):super().__init__();self.encoder=nn.Sequential(nn.BatchNorm2d(3))
    model=Model().train();freeze_encoder_bn(model)
    assert not model.encoder[0].training and model.encoder[0].weight.requires_grad
    return dict(passed=True,checks=['JS identity/symmetry/range','finite gradients through both views','empty-class stability','dedicated random-stream reproducibility','shape and spatial alignment','fixed stress views','BN statistics frozen, affine trainable'])

if __name__=='__main__':
    import json
    print(json.dumps(self_test()))
