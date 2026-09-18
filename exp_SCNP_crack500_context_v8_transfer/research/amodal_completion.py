"""Training-only bounded synthetic occlusion with amodal GT retained.

Related concepts: Random Erasing https://arxiv.org/abs/1708.04896 and
MIC (CVPR2023) https://arxiv.org/abs/2212.01322 . This independent supervised
variant uses training GT geometry for one mask center, no target-domain data
or pseudo labels, and does not claim to invent masked consistency.
"""
import torch
from torch.nn import functional as F
from chroma_consistency import colors,plain,self_test as chroma_self_test

@torch.no_grad()
def amodal_view(x,labels,geometry,generator,step,probability=.75):
    x=plain(x);rgb,mu,sd=colors(x);y=plain(labels).float();g=plain(geometry).float()
    n,_,h,w=rgb.shape;device=rgb.device
    assert y.shape==g.shape==(n,1,h,w) and 0<=probability<=1
    active=(torch.rand(n,1,1,1,device=device,generator=generator)<probability)
    mask=torch.zeros_like(y);curriculum=.5+.5*min(max(step,0)/600.,1.)
    yy=torch.arange(h,device=device)[None,None,:,None]
    xx=torch.arange(w,device=device)[None,None,None,:]
    for slot in range(2):
        if slot==0:
            weights=(g.clamp_min(0)*y).flatten(1)
            weights=torch.where(weights.sum(1,keepdim=True)>0,weights,torch.ones_like(weights))
            centers=torch.multinomial(weights,1,generator=generator).view(n)
            cy,cx=centers//w,centers%w
        else:
            cy=torch.randint(h,(n,),device=device,generator=generator)
            cx=torch.randint(w,(n,),device=device,generator=generator)
        size=torch.randint(36,101,(n,2),device=device,generator=generator).float()*curriculum
        hh=(size[:,0]*h/512).long().clamp(1,h)
        ww=(size[:,1]*w/512).long().clamp(1,w)
        top=torch.minimum((cy-hh//2).clamp_min(0),h-hh)[:,None,None,None]
        left=torch.minimum((cx-ww//2).clamp_min(0),w-ww)[:,None,None,None]
        rect=(yy>=top)&(yy<top+hh[:,None,None,None])&(xx>=left)&(xx<left+ww[:,None,None,None])
        mask=torch.maximum(mask,(rect&active).float())
    # Smooth random appearance, independent of all target/test images.
    texture=F.interpolate(torch.rand(n,3,4,4,device=device,generator=generator),size=(h,w),mode='bilinear',align_corners=False)
    alpha=.75+.25*torch.rand(n,1,1,1,device=device,generator=generator)
    opaque=torch.rand(n,1,1,1,device=device,generator=generator)<.5
    alpha=torch.where(opaque,torch.ones_like(alpha),alpha)*mask
    mixed=(rgb*(1-alpha)+texture*alpha-mu)/sd
    result=torch.where(mask.bool(),mixed,x)
    return result,mask

def self_test():
    original=chroma_self_test()
    torch.manual_seed(12)
    x=torch.randn(5,3,128,128);y=torch.zeros(5,1,128,128);y[:,:,60:68,:]=1
    geom=y.clone();before_x=x.clone();before_y=y.clone();state=torch.get_rng_state().clone()
    a,mask=amodal_view(x,y,geom,torch.Generator().manual_seed(22),600,probability=1.)
    b,other=amodal_view(x,y,geom,torch.Generator().manual_seed(22),600,probability=1.)
    assert torch.equal(a,b) and torch.equal(mask,other)
    assert torch.equal(x,before_x) and torch.equal(y,before_y)
    assert torch.equal(state,torch.get_rng_state()),'Occlusion consumed global RNG'
    assert torch.equal(a[~mask.bool().expand_as(a)],x[~mask.bool().expand_as(x)])
    assert torch.isfinite(a).all() and bool(((mask==0)|(mask==1)).all())
    fractions=mask.mean((1,2,3));assert bool(((fractions>0)&(fractions<.08)).all())
    assert bool(((mask*y).sum((1,2,3))>0).all()),'Biased mask missed foreground'
    untouched,zero=amodal_view(x,y,geom,torch.Generator().manual_seed(3),600,probability=0.)
    assert torch.equal(untouched,x) and not bool(zero.any())
    for constant in (0.,1.):
        z,m=amodal_view(x,torch.full_like(y,constant),torch.zeros_like(geom),torch.Generator().manual_seed(4),0,probability=1.)
        assert torch.isfinite(z).all() and bool(m.any())
    return dict(passed=True,checks=original['checks']+['bounded mask area below8%','GT and input immutable','unmasked pixels exactly unchanged','reproducible independent RNG','foreground-targeted coverage','empty-class fallback','disabled augmentation identity'],mask_fractions=fractions.tolist())

if __name__=='__main__':
    import json
    print(json.dumps(self_test()))
