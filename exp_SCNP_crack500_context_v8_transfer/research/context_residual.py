"""Independent unrestricted residual head; official DINOv2 remains unmodified/frozen.

Context fusion, residual learning and uncertainty gates are established concepts.
This experiment tests their utility for this task, not a claim of inventing them.
"""
import hashlib,json,os,subprocess,sys
from pathlib import Path
os.environ.setdefault('XFORMERS_DISABLED','1')
import torch
from torch import nn
from torch.nn import functional as F
from convnext_unet import PretrainedUNet as LocalUNet,WEIGHT_SHA
from pretrained_unet import Block,plain,setup_pretrained_data,ImageNetNormalize
from chroma_consistency import WARM_REL,WARM_SHA
ROOT=Path(__file__).resolve().parents[2]

def digest(module):
    h=hashlib.sha256()
    for k,v in module.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

class ContextResidualHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project=nn.Sequential(nn.Conv2d(1536,64,1,bias=False),nn.GroupNorm(8,64),nn.GELU())
        self.mix=Block(98,64);self.output=nn.Conv2d(64,1,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)
    def forward(self,local,base,context=None):
        h,w=base.shape[-2:];size=(h//4,w//4)
        p=base.float().softmax(1)
        # Zero context bypasses the projection, so its parameters receive no
        # gradients in the local-only reference. Record effective parameter count.
        if context is None:c=local.new_zeros((len(local),64,*size))
        else:
            # Undo DINO padding before fusing spatial coordinates. The final
            # 512-to128 reduction is exact 4x4 pooling, preserving alignment.
            c=F.interpolate(self.project(context),size=(h+6,w+6),mode='bilinear',align_corners=False)
            c=F.avg_pool2d(c[:,:,3:3+h,3:3+w],4)
        z=torch.cat((F.avg_pool2d(local,4),F.avg_pool2d(p,4).to(local.dtype),c),1)
        raw=F.interpolate(self.output(self.mix(z)),size=(h,w),mode='bilinear',align_corners=False).float()
        probability=p[:,1:2];gate=torch.ones_like(probability)
        delta=raw
        result=base.float()+torch.cat((-.5*delta,.5*delta),1)
        return result,delta,gate

class PretrainedUNet(nn.Module):
    def __init__(self,method='dino_context',warm_path=None,warm_sha=None):
        super().__init__();assert method in ('local_refine','dino_context');self.method=method
        self.base=LocalUNet();assert warm_path is not None and warm_sha is not None;warm=Path(warm_path)
        assert hashlib.sha256(warm.read_bytes()).hexdigest()==warm_sha
        checkpoint=torch.load(warm,map_location='cpu',weights_only=False);assert checkpoint['config']['dataset']=='crack500'
        self.base.load_state_dict(checkpoint['model'],strict=True);self.base.requires_grad_(False);self.base.eval();del checkpoint
        self.dino=None
        if method=='dino_context':
            manifest=json.loads((ROOT/'auto_res_logs/topo_free_context_v8_20260918/resource_manifest.json').read_text());repo=ROOT/'upstream_dinov2'
            assert subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()==manifest['revision']
            assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True).strip()
            path=ROOT/manifest['weight_path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==manifest['weight_sha256']
            sys.path.insert(0,str(repo))
            from dinov2.hub.backbones import dinov2_vits14
            # Backbone construction must not alter the paired training RNG.
            with torch.random.fork_rng(devices=[]):
                self.dino=dinov2_vits14(pretrained=False)
            self.dino.load_state_dict(torch.load(path,map_location='cpu',weights_only=True),strict=True)
            self.dino.requires_grad_(False);self.dino.eval()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(271828);self.head=ContextResidualHead()
        if method=='local_refine':self.head.project.requires_grad_(False)
        self.identity_verified=False;self.last_diagnostic={}
    @property
    def encoder(self):return self.base.encoder
    def train(self,mode=True):
        super().train(mode);self.base.eval()
        if self.dino is not None:self.dino.eval()
        return self
    def forward(self,image):
        x=plain(image)
        with torch.no_grad():
            z=x;stages=[]
            for index,layer in enumerate(self.base.encoder.features):
                z=layer(z)
                if index in (1,3,5,7):stages.append(z)
            a,b,c,z=stages;half=self.base.half_res(F.avg_pool2d(x,2))
            for block,skip in zip(self.base.decoders,[c,b,a,half,self.base.shallow(x)]):
                z=block(torch.cat((F.interpolate(z,size=skip.shape[-2:],mode='bilinear',align_corners=False),skip),1))
            base=self.base.output(z);context=None
            if self.dino is not None:
                assert x.shape[-2:]==(512,512)
                # Symmetric zero-padding in normalized space, no lost boundary pixels.
                padded=F.pad(x,(3,3,3,3),value=0)
                intermediate=self.dino.get_intermediate_layers(padded,n=4,reshape=True,norm=True)
                context=torch.cat(intermediate,1)
                assert context.shape[1:]==(1536,37,37)
        result,delta,gate=self.head(z,base,context)
        if not self.identity_verified and not bool(torch.count_nonzero(self.head.output.weight)) and not bool(torch.count_nonzero(self.head.output.bias)):
            assert torch.equal(result,base.float()),'Zero initialization changed the base output'
            with torch.no_grad():original=self.base(x)
            assert torch.equal(original,base),'Independent frozen feature route differs'
            self.identity_verified=True
        self.last_diagnostic={'mean_abs_delta':float(delta.detach().abs().mean()),'max_abs_delta':float(delta.detach().abs().max()),'mean_gate':float(gate.detach().mean()),'context_active':self.dino is not None}
        return result

def trainable_hash(model):return digest(model.head)
def verify_frozen(model):
    assert all(not p.requires_grad for p in model.base.parameters())
    if model.dino is not None:assert all(not p.requires_grad for p in model.dino.parameters())
    return {'base':digest(model.base),'dino':digest(model.dino) if model.dino is not None else None}
def make_optimizer(model):
    parameters=list(model.head.parameters()) if model.method=='dino_context' else list(model.head.mix.parameters())+list(model.head.output.parameters())
    return torch.optim.AdamW(parameters,lr=3e-4,weight_decay=.01)

if __name__=='__main__':
    torch.set_num_threads(2);torch.manual_seed(123)
    t=torch.randn(2,3,32,32,requires_grad=True)
    assert torch.equal(F.avg_pool2d(t,4),F.adaptive_avg_pool2d(t,(8,8)))
    h=ContextResidualHead();z=torch.randn(2,32,32,32);base=torch.randn(2,2,32,32);c=torch.randn(2,1536,3,3)
    zero,delta,gate=h(z,base,c);assert torch.equal(zero,base) and torch.count_nonzero(delta)==0
    loss=F.cross_entropy(zero,torch.randint(0,2,(2,32,32)));loss.backward()
    assert h.output.weight.grad.abs().sum()>0
    opt=torch.optim.SGD(h.parameters(),lr=.01);opt.step();opt.zero_grad()
    result,delta,gate=h(z,base,c);F.cross_entropy(result,torch.randint(0,2,(2,32,32))).backward()
    assert h.project[0].weight.grad.abs().sum()>0 and all(torch.isfinite(p.grad).all() for p in h.parameters() if p.grad is not None)
    with torch.no_grad():h.output.bias.fill_(100)
    _,delta,gate=h(z,base,None);assert delta.abs().max()>4 and torch.isfinite(delta).all() and torch.equal(gate,torch.ones_like(gate))
    print(json.dumps({'passed':True,'zero_init_identity':True,'output_gradient':True,'context_gradient_after_update':True,'unrestricted_finite_correction':True,'inference_requires_no_target':True}))
