"""Engineering checks for the changed baseline, not accuracy selection."""
import json,time
from pathlib import Path
import numpy as np
import torch
from pretrained_unet import PretrainedUNet,ImageNetNormalize,setup_pretrained_data,ROOT,WEIGHT_SHA

torch.set_num_threads(2);torch.manual_seed(0);checks=[]
m=PretrainedUNet();official=torch.load(ROOT/'data/pretrained_weights/resnet34-b627a593.pth',weights_only=True,map_location='cpu')
missing_counters=[]
for name,value in m.encoder.state_dict().items():
    if name in official:assert torch.equal(value,official[name]),name
    else:
        assert name.endswith('.num_batches_tracked') and value.numel()==1 and value.item()==0,name
        missing_counters.append(name)
checks.append('All retained checkpoint tensors match; only historical absent BN counters default to zero')
raw=torch.tensor([0.,127.5,255.])[:,None,None].expand(3,5,7).clone()
z=ImageNetNormalize(True)({'image':raw})['image'];expected=(raw/255-torch.tensor([.485,.456,.406])[:,None,None])/torch.tensor([.229,.224,.225])[:,None,None]
assert torch.equal(z,expected);checks.append('Explicit ImageNet channel normalization matches independent formula')
x=torch.randn(2,3,63,65);y=m(x);assert y.shape==(2,2,63,65) and torch.isfinite(y).all()
y.square().mean().backward()
assert m.encoder.conv1.weight.grad.abs().sum()>0 and m.output.weight.grad.abs().sum()>0
assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
checks.append('Odd input dimensions preserved; finite backward reaches pretrained encoder and decoder')
tr1,va1,man1,_=setup_pretrained_data(3);tr2,va2,man2,_=setup_pretrained_data(3)
assert man1==man2
for i in (0,1):
    # The pinned author's RandHue also consumes the worker Torch RNG.
    state=torch.get_rng_state().clone();a=tr1[i];torch.set_rng_state(state);b=tr2[i]
    assert torch.equal(a['image'],b['image']) and torch.equal(a['label'],b['label'])
    assert set(torch.unique(a['label']).tolist()).issubset({0.,1.})
    assert a['image'].shape==(3,512,512)
assert torch.equal(va1[0]['image'],va2[0]['image'])
checks.append('Paired transforms reproducible with same MONAI and worker Torch RNG; binary labels and full resolution retained')
report={'created':time.time(),'passed':True,'checks':checks,'pretrained_sha256':WEIGHT_SHA,'historical_missing_bn_counters':missing_counters}
(Path(__file__).resolve().parents[1]/'auto_res_logs/results/pretrained_tests.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
