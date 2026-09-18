"""Visibility-conditioned SCNP backward budgeting; forward logits unchanged.

Counts how often each source logit is selected by official same-class 3x3
pooling. Only occluded source pixels plus their one-pixel halo are scaled.
This is a detached gradient transform, not the exact gradient of a new scalar
objective. Original SCNP already describes repeated hard-neighbor penalties.
"""
import torch
from torch.nn import functional as F
from chroma_consistency import plain
from losses import one_hot,scnp_route,load_upstream_loss
from amodal_completion import self_test as amodal_self_test

def budgeted_logits(logits,labels,geometry,occlusion,strength=1.):
    z=plain(logits).float();y=plain(labels).float();g=plain(geometry).float();mask=plain(occlusion).float()
    assert z.ndim==4 and y.shape==g.shape==mask.shape==(len(z),1,*z.shape[-2:]) and 0<=strength<=1
    with torch.no_grad():
        target=one_hot(y,z.shape[1]).to(z)
        _,indices=scnp_route(z.detach(),target,3)
        flat=indices.flatten(2)
        counts=torch.zeros_like(flat).scatter_add_(2,flat,torch.ones_like(flat)).reshape_as(z).float()
        halo=F.max_pool2d(mask,3,1,1)>0
        cap=2.+4.*g.clamp(0,1)
        bounded=(cap/counts.clamp_min(1)).clamp(max=1.)
        scale=torch.where(halo,1.-strength*(1.-bounded),torch.ones_like(bounded))
        diag={'active':True,'strength':strength,'count_max':float(counts.max()),'scale_min':float(scale.min()),'scaled_logit_fraction':float((scale<1).float().mean()),'occlusion_halo_fraction':float(halo.float().mean())}
    return z.detach()+scale*(z-z.detach()),diag

def self_test():
    prior=amodal_self_test();torch.manual_seed(17)
    z=torch.randn(2,2,19,23,requires_grad=True);y=torch.randint(0,2,(2,1,19,23)).float();g=torch.zeros_like(y);mask=torch.zeros_like(y);mask[:,:,5:14,5:16]=1
    loss=load_upstream_loss();base=loss(z,y[:,0].long());raw=torch.autograd.grad(base,z)[0]
    guarded,d=budgeted_logits(z,y,g,mask,1.);torch.testing.assert_close(guarded,z,rtol=0,atol=0)
    candidate=loss(guarded,y[:,0].long());torch.testing.assert_close(candidate,base,rtol=0,atol=0)
    grad=torch.autograd.grad(candidate,z,retain_graph=True)[0]
    # Independent derivative of the transform verifies scaling and locality.
    scale=torch.autograd.grad(guarded.sum(),z)[0];torch.testing.assert_close(grad,raw*scale,rtol=1e-5,atol=1e-8)
    outside=(F.max_pool2d(mask,3,1,1)==0).expand_as(z);torch.testing.assert_close(grad[outside],raw[outside],rtol=0,atol=0)
    assert d['scale_min']>=2/9-1e-6 and d['scaled_logit_fraction']>0 and torch.isfinite(grad).all()
    for strength,region in [(0.,mask),(1.,torch.zeros_like(mask))]:
        q,_=budgeted_logits(z,y,g,region,strength);gg=torch.autograd.grad(loss(q,y[:,0].long()),z)[0];torch.testing.assert_close(gg,raw,rtol=0,atol=0)
    for target in (torch.zeros_like(y),torch.ones_like(y)):
        q,_=budgeted_logits(z,target,g,mask,1.);assert torch.isfinite(loss(q,target[:,0].long()))
    return dict(passed=True,checks=prior['checks']+['exact unchanged SCNP forward loss','known backward scale','visible logits gradient identical','scale bounded below2/9','disabled and no-occlusion equivalence','empty-class stability'],budget_diagnostic=d)

if __name__=='__main__':
    import json
    print(json.dumps(self_test()))
