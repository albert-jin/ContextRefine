"""Research candidate: allocation by marginal completion of local same-class paths.

No claim of first path/topology loss. Probability products are a local proxy,
not a proof of global connectivity or independent calibrated probabilities.
"""
import torch
from torch.nn import functional as F

@torch.no_grad()
def completion_utility(correct_probability,target,lengths=(3,5,9)):
    p=correct_probability.clamp(1e-5,1-1e-5);logs=p.log()
    total=torch.zeros_like(p);count=torch.zeros_like(p)
    for length in lengths:
        ones=torch.ones((1,1,1,length),device=p.device,dtype=p.dtype)
        diagonal=torch.eye(length,device=p.device,dtype=p.dtype)[None,None]
        for kernel in (ones,ones.transpose(-2,-1),diagonal,diagonal.flip(-1)):
            pad=(kernel.shape[-2]//2,kernel.shape[-1]//2)
            fg=F.conv2d(target,kernel,padding=pad)
            real=F.conv2d(torch.ones_like(target),kernel,padding=pad)
            valid=(((fg<.1)|(fg>length-.1))&(real>length-.1)).to(p)
            products=F.conv2d(logs,kernel,padding=pad).exp()*valid
            total+=F.conv_transpose2d(products,kernel,padding=pad)
            count+=F.conv_transpose2d(valid,kernel,padding=pad)
    # At source i: average over paths P containing i of (1-p_i)*prod(j!=i,p_j).
    return ((1-p)*total/(p*count.clamp_min(1))).clamp(0,1)

def conditional_route(margin,target,mode='conditional',strength=8.):
    from losses import scnp_route
    from innovations import normalize_class
    routed,indices=scnp_route(margin,target)
    with torch.no_grad():
        idx=indices.flatten(2);counts=torch.zeros_like(margin).flatten(2)
        counts.scatter_add_(2,idx,torch.ones_like(idx,dtype=margin.dtype))
        correct=margin.sigmoid()*target+(1-margin.sigmoid())*(1-target)
        if mode=='conditional':u=completion_utility(correct,target)
        elif mode=='conditional_error':u=1-correct
        elif mode=='conditional_uniform':u=torch.zeros_like(margin)
        else:raise ValueError(mode)
        # Counteract source reuse, then preserve total auxiliary mass per class.
        source=(1+strength*u.flatten(2))/counts.clamp_min(1)
        weight=normalize_class(source.gather(2,idx).reshape_as(margin),target)
    loss=(weight*F.binary_cross_entropy_with_logits(routed,target,reduction='none')).mean()
    return loss,{'mean_utility':float(u.mean()),'foreground_utility':float((u*target).sum()/target.sum().clamp_min(1)),
                 'weight_max':float(weight.max()),'mean_weight':float(weight.mean())}

def test_completion():
    # Isolated bottleneck should receive more completion credit than one inside a broken path.
    target=torch.ones(1,1,3,3);a=torch.full_like(target,.99);a[0,0,1,1]=.1
    b=a.clone();b[0,0,1,:]=.1;b[0,0,:,1]=.1;b[0,0,0,0]=.1;b[0,0,2,2]=.1;b[0,0,0,2]=.1;b[0,0,2,0]=.1
    ua=completion_utility(a,target,(3,));ub=completion_utility(b,target,(3,))
    assert ua[0,0,1,1]>.8 and ub[0,0,1,1]<.02,(ua,ub)
    # Single-pixel image has no length-3 path and must not invent padded support.
    assert completion_utility(torch.ones(1,1,1,1)*.1,torch.ones(1,1,1,1),(3,)).item()==0
    # Class inversion leaves correct probability/path membership unchanged.
    torch.testing.assert_close(completion_utility(a,target,(3,)),completion_utility(a,1-target,(3,)))
    print('PASS: isolated repair credit, invalid padding, class symmetry')

if __name__=='__main__':test_completion()
