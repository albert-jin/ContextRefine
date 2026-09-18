"""SCNP forward-preserving, per-image/class/channel gradient-mass allocation."""
import torch
from torch.nn import functional as F

class _MassAllocation(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weights,target,alpha,diag):
        ctx.save_for_backward(weights,target);ctx.alpha=alpha;ctx.diag=diag
        return x.view_as(x)
    @staticmethod
    def backward(ctx,gradient):
        weights,target=ctx.saved_tensors
        if ctx.alpha==0:return gradient,None,None,None,None
        weighted=gradient*weights;scale=torch.zeros_like(gradient);errors=[]
        for mask in (target,1-target):
            old=(gradient.abs()*mask).sum((-2,-1),keepdim=True)
            new=(weighted.abs()*mask).sum((-2,-1),keepdim=True)
            factor=torch.where(old>0,old/new.clamp_min(torch.finfo(gradient.dtype).tiny),torch.ones_like(old))
            scale+=mask*factor
        result=(1-ctx.alpha)*gradient+ctx.alpha*weighted*scale
        for mask in (target,1-target):
            old=(gradient.abs()*mask).sum((-2,-1),keepdim=True)
            new=(result.abs()*mask).sum((-2,-1),keepdim=True)
            errors.append(((old-new).abs()/old.clamp_min(1e-15)).max())
        ctx.diag['gradient_mass_max_relative_error']=float(torch.stack(errors).max())
        return result,None,None,None,None

def allocate_logits(logits,labels,geometry,method,step,steps):
    from losses import scnp_route
    from conditional_budget import completion_utility
    alpha=max(0.,min(1.,(step-.25*steps)/(.25*steps)))
    target=F.one_hot(labels.long(),2).movedim(-1,1).to(logits)
    diag={'allocation_alpha':alpha}
    if alpha==0:return logits,diag
    with torch.no_grad():
        _,indices=scnp_route(logits,target);idx=indices.flatten(2)
        counts=torch.zeros_like(logits).flatten(2);counts.scatter_add_(2,idx,torch.ones_like(idx,dtype=logits.dtype));counts=counts.reshape_as(logits)
        correct=(logits.softmax(1)*target).sum(1,keepdim=True)
        if method=='mass_uniform':utility=torch.zeros_like(correct)
        elif method=='mass_error':utility=1-correct
        elif method=='mass_geometry':utility=geometry*(1-correct)
        elif method=='mass_completion':utility=completion_utility(correct,target[:,1:2])
        else:raise ValueError(method)
        weights=(1+8*utility)/counts.clamp_min(1)
        diag.update(mean_utility=float(utility.mean()),reused_source_fraction=float((counts>1).float().mean()))
    return _MassAllocation.apply(logits,weights,target,alpha,diag),diag

def test_mass():
    torch.manual_seed(5);torch.set_num_threads(2)
    for dtype in (torch.float32,torch.float64):
        x=torch.randn(2,2,11,13,dtype=dtype,requires_grad=True);w=torch.rand_like(x)*4+.1;y=(torch.rand_like(x)>.5).to(x);g=torch.randn_like(x)
        diag={};z=_MassAllocation.apply(x,w,y,.73,diag);assert torch.equal(x,z)
        new=torch.autograd.grad((z*g).sum(),x)[0]
        for mask in (y,1-y):torch.testing.assert_close((new.abs()*mask).sum((-2,-1)),(g.abs()*mask).sum((-2,-1)),rtol=2e-6,atol=1e-10)
        assert torch.equal(new.sign(),g.sign())
    print('PASS: identical forward, exact grouped L1 gradient mass, coordinate signs')
if __name__=='__main__':test_mass()
