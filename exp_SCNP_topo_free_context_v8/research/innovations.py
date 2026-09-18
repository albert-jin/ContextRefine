"""Exploratory SCNP variants, with fixed controls. No novelty or topology guarantee."""
import torch
from torch.nn import functional as F
from losses import scnp_route

def class_mean(value,target):
    fg=(value*target).sum()/target.sum().clamp_min(1)
    bg=(value*(1-target)).sum()/(1-target).sum().clamp_min(1)
    return .5*(fg+bg)

def normalize_class(weights,target):
    # Detached allocation: every class keeps its original total auxiliary mass.
    mean_fg=(weights*target).sum()/target.sum().clamp_min(1)
    mean_bg=(weights*(1-target)).sum()/(1-target).sum().clamp_min(1)
    return weights/(mean_fg*target+mean_bg*(1-target)).clamp_min(1e-6)

def weighted_route(margin,target,priority,mode):
    routed,indices=scnp_route(margin,target)
    with torch.no_grad():
        idx=indices.flatten(2);counts=torch.zeros_like(margin).flatten(2)
        counts.scatter_add_(2,idx,torch.ones_like(idx,dtype=margin.dtype))
        if mode=='uniform':utility=torch.zeros_like(margin)
        elif mode=='uncertainty':utility=1-(2*margin.sigmoid()-1).abs()
        else:
            confidence=margin.sigmoid()*target+(1-margin.sigmoid())*(1-target)
            utility=priority*(1-confidence)
        source=((2+8*utility.flatten(2))/counts.clamp_min(1)).clamp_max(1)
        weight=source.gather(2,idx).reshape_as(margin)
        weight=normalize_class(weight,target)
    loss=(weight*F.binary_cross_entropy_with_logits(routed,target,reduction='none')).mean()
    return loss,{'mean_weight':float(weight.mean()),'utility_fraction':float((utility>0).float().mean())}

def soft_tail(margin,target,temperature=.5):
    # Log-mean-exp of same-class per-source CE, a smooth upper-tail risk.
    ce=F.binary_cross_entropy_with_logits(margin,target,reduction='none')
    bs,_,h,w=margin.shape
    neighbors=F.unfold(ce,3,padding=1).view(bs,9,h,w)
    classes=F.unfold(target,3,padding=1).view(bs,9,h,w)
    real=F.unfold(torch.ones_like(target),3,padding=1).view(bs,9,h,w)>0
    valid=(classes==target)&real
    scores=(neighbors/temperature).masked_fill(~valid,-1e6)
    risk=temperature*(torch.logsumexp(scores,dim=1,keepdim=True)-valid.sum(1,keepdim=True).float().log())
    return risk.mean()

def path_risk(margin,target,balanced=True):
    # Only whole same-class paths qualify. Two axial orientations, lengths 3/5/9.
    terms=[];stats=[]
    for length in (3,5,9):
        for kernel in ((1,length),(length,1)):
            padding=tuple(k//2 for k in kernel)
            minimum=-F.max_pool2d(-margin,kernel,1,padding)
            maximum=F.max_pool2d(margin,kernel,1,padding)
            # Padded positions excluded; no supervision invented outside the crop.
            counts=F.avg_pool2d(target,kernel,1,padding,count_include_pad=True)*length
            real=F.avg_pool2d(torch.ones_like(target),kernel,1,padding,count_include_pad=True)*length
            fg=(counts>=length-.1)&(real>=length-.1)
            bg=(counts<.1)&(real>=length-.1)
            positive=F.softplus(-minimum);negative=F.softplus(maximum)
            if balanced:
                term=.5*((positive*fg).sum()/fg.sum().clamp_min(1)+(negative*bg).sum()/bg.sum().clamp_min(1))
            else:term=((positive*fg).sum()+(negative*bg).sum())/(fg.sum()+bg.sum()).clamp_min(1)
            terms.append(term);stats.append(float(fg.float().mean()))
    return torch.stack(terms).mean(),{'valid_fg_path_fraction':sum(stats)/len(stats)}

def soft_erode(x):
    return torch.minimum(-F.max_pool2d(-x,(3,1),1,(1,0)),-F.max_pool2d(-x,(1,3),1,(0,1)))

def soft_skeleton(x,iterations=10):
    eroded=soft_erode(x);opened=F.max_pool2d(eroded,3,1,1);skel=F.relu(x-opened)
    for _ in range(iterations):
        x=eroded;eroded=soft_erode(x);opened=F.max_pool2d(eroded,3,1,1)
        delta=F.relu(x-opened);skel=skel+F.relu(delta-skel*delta)
    return skel

def cldice_loss(margin,target):
    # Known clDice-style baseline, never claimed as our innovation.
    p=margin.sigmoid();sp=soft_skeleton(p);sy=soft_skeleton(target)
    axes=(1,2,3);precision=((sp*target).sum(axes)+1e-5)/(sp.sum(axes)+1e-5)
    recall=((sy*p).sum(axes)+1e-5)/(sy.sum(axes)+1e-5)
    return (1-2*precision*recall/(precision+recall+1e-5)).mean()

def auxiliary(method,margin,target,priority):
    if method in ('margin_joint','balanced_joint'):
        routed,_=scnp_route(margin,target)
        ce=F.binary_cross_entropy_with_logits(routed,target,reduction='none')
        return (class_mean(ce,target) if method=='balanced_joint' else ce.mean()),{}
    if method in ('uniform','uncertainty','geometry'):
        return weighted_route(margin,target,priority,method)
    if method=='softtail':return soft_tail(margin,target),{}
    if method in ('path','path_unbalanced'):
        return path_risk(margin,target,balanced=method=='path')
    if method=='cldice':return cldice_loss(margin,target),{}
    raise ValueError(method)
