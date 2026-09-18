"""Check allocation invariants and source selection against hand-computable cases."""
import torch
from innovations import normalize_class,soft_tail,path_risk,weighted_route
torch.manual_seed(9)
y=(torch.rand(2,1,13,13)>.6).float();z=torch.randn_like(y,requires_grad=True)
w=normalize_class(torch.rand_like(y)+.01,y)
assert torch.allclose((w*y).sum(),y.sum(),atol=1e-4)
assert torch.allclose((w*(1-y)).sum(),(1-y).sum(),atol=1e-4)
constant=torch.full((1,1,7,7),2.,requires_grad=True);ones=torch.ones_like(constant)
assert torch.allclose(soft_tail(constant,ones),torch.nn.functional.softplus(-constant).mean(),atol=1e-6)
for method in ('uniform','geometry','uncertainty'):
    loss,diag=weighted_route(z,y,torch.rand_like(y),method);grad=torch.autograd.grad(loss,z,retain_graph=True)[0]
    assert torch.isfinite(grad).all() and grad.abs().sum()>0
    assert abs(diag['mean_weight']-1)<1e-5
loss,_=path_risk(z,y);loss.backward();assert torch.isfinite(z.grad).all()
# Correct constant predictions must have less path risk than wrong ones.
a,_=path_risk(torch.full((1,1,13,13),3.),torch.ones(1,1,13,13))
b,_=path_risk(torch.full((1,1,13,13),-3.),torch.ones(1,1,13,13));assert a<b
print('PASS: class budget mass, constant risk, finite gradients, path direction')
