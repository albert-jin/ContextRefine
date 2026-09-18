"""Full-budget DynUNet recipe derived from the pinned TopoMortar reference."""
import argparse,ast,hashlib,importlib.util,json,os,random,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
import torch,yaml,monai
from torch.nn import functional as F
from monai.networks.nets import DynUNet
from monai.networks import one_hot
from monai.transforms import Compose,NormalizeIntensityd
from monai.data import Dataset,DataLoader,worker_init_fn
from scipy.ndimage import distance_transform_edt,maximum_filter
from skimage.morphology import skeletonize
from losses import load_upstream_loss
from mass_budget import allocate_logits
from train import metrics
REPO=Path(__file__).resolve().parents[1];ROOT=REPO.parent
def atomic(p,obj):
    q=p.with_suffix('.tmp');q.write_text(json.dumps(obj,indent=2));q.replace(p)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def reference_losses():
    path=REPO/'reference/lib/losses.py';tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('_dice_loss','_crossentropy_loss','CEDiceLoss','clDiceLoss')]
    ns={'torch':torch,'_Loss':torch.nn.modules.loss._Loss,'one_hot':one_hot,'utils_cldice':module(REPO/'reference/lib/losses_cldice.py','ref_cldice')}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns['CEDiceLoss'](),ns['clDiceLoss'](iterations=3,alpha=.5)
def setup_data(seed,augment=True):
    cfg=yaml.safe_load((REPO/'reference/config/template_topomortar.yaml').read_text())
    rows={};manifest={}
    splits=yaml.safe_load((ROOT/'data/TopoMortar/dataset/splits.yaml').read_text())[1]
    for split in ('train','val'):
        rows[split]=[];manifest[split]=[]
        for rel in splits[split]:
            xp=ROOT/'data/TopoMortar'/rel;yp=Path(str(xp).replace('/images/','/accurate/'))
            x=np.asarray(Image.open(xp).convert('RGB'),dtype=np.float32).transpose(2,0,1).copy()
            y=(np.asarray(Image.open(yp))>0).astype(np.float32)[None]
            assert x.shape==(3,512,512) and y.shape==(1,512,512)
            geom=np.zeros((512,512),np.float32)
            for region in (y[0]>0,y[0]==0):
                geom+=maximum_filter(skeletonize(region).astype(np.float32)/(1+distance_transform_edt(region)),size=3)*region
            geom[[0,-1],:]=0;geom[:,[0,-1]]=0
            rows[split].append({'image':x,'label':y,'geom':geom[None],'id':xp.stem})
            manifest[split].append({'id':xp.stem,'image_sha256':sha(xp),'mask_sha256':sha(yp)})
    assert not ({r['image_sha256'] for r in manifest['train']}&{r['image_sha256'] for r in manifest['val']})
    transforms=[]
    for spec in cfg['transform_train'][3:]:
        name=spec['name'].split('.')[-1];params=dict(spec['params'])
        # Warp precomputed geometry together with the label; unchanged for all controls.
        if name in ('RandRotated','RandZoomd','RandAxisFlipd'):
            params['keys']=['image','label','geom']
            if 'mode' in params:params['mode']=list(params['mode'])+[params['mode'][-1]]
        transforms.append(getattr(monai.transforms,name)(**params))
    # Author-provided RandHue is an existing augmentation, common to every arm.
    hue=module(REPO/'reference/lib/transforms.py','ref_transforms').RandHued(['image'],prob=.5)
    transforms.insert(1,hue)
    tr=Compose(transforms if augment else [NormalizeIntensityd(['image'],channel_wise=True)]);tr.set_random_state(seed=seed)
    val=Compose([NormalizeIntensityd(['image'],channel_wise=True)])
    return Dataset(rows['train'],tr),Dataset(rows['val'],val),manifest,cfg
def weights_hash(model):
    h=hashlib.sha256()
    for key,v in model.state_dict().items():h.update(key.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def deep_supervision_loss(pred,loss_at):
    """Python scalars preserve MetaTensor autograd; NumPy-left multiplication does not."""
    assert pred.ndim==5 and pred.shape[1]==7,pred.shape
    raw=[1.0/(2**i) for i in range(pred.shape[1])];normalizer=sum(raw)
    loss=loss_at(pred[:,0],0)
    for i in range(1,pred.shape[1]):loss=loss+(raw[i]/normalizer)*loss_at(pred[:,i],i)
    return loss/pred.shape[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['base','scnp','cldice','mass_geometry'],required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=12000)
    p.add_argument('--batch',type=int,default=10);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--val-every',type=int,default=1200);p.add_argument('--run-id',required=True)
    a=p.parse_args();out=REPO/'auto_res_logs/runs'/a.run_id;out.mkdir(parents=True,exist_ok=False)
    state={'status':'starting','pid':os.getpid(),'created':time.time()};atomic(out/'status.json',state)
    try:
        assert os.environ.get('SCNP_GPU_GUARDED')=='1' and torch.cuda.is_available()
        torch.set_num_threads(4);monai.utils.set_determinism(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        tr,va,manifest,reference=setup_data(a.seed);atomic(out/'data_manifest.json',manifest)
        g=torch.Generator().manual_seed(a.seed+12345)
        loader=DataLoader(tr,batch_size=a.batch,shuffle=True,drop_last=True,num_workers=a.workers,
                          persistent_workers=a.workers>0,worker_init_fn=worker_init_fn,generator=g,pin_memory=True)
        cfg=vars(a)|{'dataset':'topomortar','model':'MONAI DynUNet official eight-level configuration','reference_commit':'b1f31f20a41b20aa775c21078b2c432c6dcc26b4',
                      'normalization':'per-image per-channel mean/std','augmentation':'official template plus official RandHue(prob=.5) common to all',
                      'geometry':'precomputed GT skeleton-width priority warped by nearest interpolation with target',
                      'optimizer':'SGD lr.01 momentum.99 nesterov weight_decay3e-5','scheduler':'polynomial .9','clip_norm':12,
                      'precision':'FP16 autocast+GradScaler; losses/source allocation FP32','deep_supervision':'exact reference weights/main convention, seven heads',
                      'implementation_revision':'v2_python_scalar_auxiliary_weights_all7_heads_audited',
                      'threshold':.5,'selection':'best validation mean foreground Dice, also report fixed final','test_accessed':False,'size':512}
        atomic(out/'config.json',cfg)
        for source in list((REPO/'research').glob('*.py'))+list((REPO/'reference').rglob('*.py'))+list((REPO/'reference/config').glob('*.yaml')):
            dest=out/'source_snapshot'/source.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        hashes={str(f.relative_to(REPO)):sha(f) for f in (REPO/'research').glob('*.py')};atomic(out/'source_hashes.json',hashes)
        model=DynUNet(**reference['model']['params']).cuda()
        atomic(out/'initialization.json',{'parameter_sha256':weights_hash(model),'parameters':sum(p.numel() for p in model.parameters())})
        (out/'model.txt').write_text(str(model))
        opt=torch.optim.SGD(model.parameters(),lr=.01,momentum=.99,nesterov=True,weight_decay=3e-5)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda it:(1-it/a.steps)**.9)
        scaler=torch.amp.GradScaler('cuda');base,cldice=reference_losses();scnp=load_upstream_loss().cuda()
        best=-1.;start=time.time();iterator=iter(loader);torch.cuda.reset_peak_memory_stats()
        def checkpoint(step):
            return {'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'step':step,
                    'config':cfg,'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_python':random.getstate(),
                    'rng_numpy':np.random.get_state(),'loader_generator':g.get_state(),'resume_note':'Worker augmentation streams are not fully restorable; restarting is not exact continuation'}
        for step in range(1,a.steps+1):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            x=batch['image'].cuda(non_blocking=True);y=batch['label'].cuda(non_blocking=True);geom=batch['geom'].cuda(non_blocking=True)
            model.train();opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.float16):pred=model(x)
            diags=[];head_diags=[{} for _ in range(pred.shape[1])]
            audit_heads=step==1 or step%25==0 or step==a.steps
            def loss_at(z,head):
                z=z.float()
                if audit_heads:
                    def note_gradient(gradient):
                        head_diags[head].update(backward_observed=True,scaled_gradient_l1=float(gradient.detach().abs().sum()),finite=bool(torch.isfinite(gradient).all()))
                    z.register_hook(note_gradient)
                if a.method=='base':return base(z,y)
                if a.method=='cldice':return cldice(z,y)
                if a.method=='mass_geometry':
                    z,d=allocate_logits(z,y[:,0].long(),geom,a.method,step,a.steps);diags.append(d)
                return scnp(z,y[:,0].long())
            loss=deep_supervision_loss(pred,loss_at)
            assert torch.isfinite(loss),'nonfinite loss'
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),12)
            if audit_heads:
                assert len(head_diags)==7 and all(d.get('backward_observed') for d in head_diags),'Missing deep-supervision backward path'
                if step==1:assert all(d['finite'] and d['scaled_gradient_l1']>0 for d in head_diags),'Invalid initial head gradients'
            scaler.step(opt);scaler.update();scheduler.step()
            if step==a.steps//4:atomic(out/'warmup_end_parameters.json',{'step':step,'parameter_sha256':weights_hash(model)})
            if step==1 or step%25==0 or step==a.steps:
                row={'step':step,'loss':float(loss.detach()),'lr':opt.param_groups[0]['lr'],'elapsed_sec':time.time()-start,
                     'sample_ids':list(batch['id']),'grad_diags':diags,'head_grad_diags':head_diags,'scale':scaler.get_scale(),
                     'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                state.update(status='running',step=step,elapsed_sec=row['elapsed_sec']);atomic(out/'status.json',state);print(json.dumps(row),flush=True)
                smi=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
                for line in smi.splitlines():
                    uuid,total,used=[s.strip() for s in line.split(',')]
                    if uuid==os.environ['CUDA_VISIBLE_DEVICES'] and int(used)/int(total)>=.8:
                        torch.save(checkpoint(step),out/'memory_stop.pt');raise RuntimeError('GPU reached80%; own task stopped')
            if step%a.val_every==0 or step==a.steps:
                model.eval();rows=[]
                with torch.no_grad():
                    for item in va:
                        with torch.autocast('cuda',dtype=torch.float16):z=model(item['image'][None].cuda())
                        prediction=(z.float().softmax(1)[0,1]>=.5).cpu().numpy()
                        rows.append({'id':item['id']}|metrics(prediction,np.asarray(item['label'][0])>0))
                mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='id'};value={'step':step,'mean':mean,'per_image':rows}
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
                if mean['dice']>best:
                    best=mean['dice'];torch.save(checkpoint(step),out/'best.pt');atomic(out/'best_validation.json',value)
                torch.save(checkpoint(step),out/'last.pt');print(json.dumps({'step':step,'validation':mean}),flush=True)
        atomic(out/'memory_profile.json',{'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2})
        state.update(status='completed',step=a.steps,best_dice=best,elapsed_sec=time.time()-start);atomic(out/'status.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));atomic(out/'status.json',state);(out/'traceback.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
