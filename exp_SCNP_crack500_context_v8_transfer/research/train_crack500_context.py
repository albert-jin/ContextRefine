"""Crack500 transfer of the frozen-backbone unrestricted context residual method."""
import argparse, hashlib, json, math, os, random, shutil, time, traceback
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from crack500_data import TrainingData, seed_worker, evaluate, gpu_guard
from convnext_unet import PretrainedUNet as LocalUNet, WEIGHT_SHA
from context_residual import PretrainedUNet as ContextUNet, verify_frozen, trainable_hash, digest
from chroma_consistency import paired_view, structure_js
from amodal_completion import amodal_view
from losses import load_upstream_loss

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO.parent
PHASE = ROOT / 'auto_res_logs/crack500_context_v8_20260918'
MANIFEST_SHA = '8044bc6505e0a82b17faa538bb718b3d3df27d4a99586dbc4eefebdef0d79ddb'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def atomic(p,d):
    p=Path(p); t=p.with_suffix('.tmp'); t.write_text(json.dumps(d,indent=2)); t.replace(p)

def model_for(stage,warm):
    if stage=='base': return LocalUNet()
    warm=Path(warm); lock=read(warm.parent/'test_lock.json')
    assert sha(warm)==lock['checkpoint_sha256']
    assert read(warm.parent/'config.json')['dataset']=='crack500'
    return ContextUNet(stage,warm_path=warm,warm_sha=lock['checkpoint_sha256'])

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',choices=['base','local_refine','dino_context'],required=True)
    p.add_argument('--run-id',required=True); p.add_argument('--warm')
    p.add_argument('--steps',type=int,required=True); p.add_argument('--batch',type=int,default=8)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--val-every',type=int,required=True)
    p.add_argument('--seed',type=int,default=0); p.add_argument('--smoke',action='store_true')
    p.add_argument('--mode',choices=['train','test'],default='train'); a=p.parse_args()
    assert os.environ.get('SCNP_GPU_GUARDED')=='1' and torch.cuda.is_available(); gpu_guard()
    torch.set_num_threads(4); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed)
    torch.use_deterministic_algorithms(True); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    manifest=ROOT/'data/icassp/crack500/manifest.json'; assert sha(manifest)==MANIFEST_SHA
    data=read(manifest); assert data['status']=='ready'
    assert [len(data['splits'][s]) for s in ['train','val','test']]==[243,49,200]
    out=REPO/'auto_res_logs/runs'/a.run_id
    if a.mode=='test':
        assert not a.smoke
        lock=read(out/'test_lock.json'); assert sha(out/'best.pt')==lock['checkpoint_sha256']
        assert lock['manifest_sha256']==MANIFEST_SHA
        for rel,expected in lock['source_hashes'].items(): assert sha(REPO/rel)==expected,rel
        model=model_for(a.stage,a.warm).cuda(); ckpt=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(ckpt['model'],strict=True); dest=out/'test'; dest.mkdir(exist_ok=False)
        # Replay the entire validation set before accessing any test image.
        replay=evaluate(model,data['splits']['val']); expected=read(out/'best_validation.json')
        old={r['id']:r for r in expected['per_image']}
        max_error=max(abs(r['dice']-old[r['id']]['dice']) for r in replay['per_image'])
        atomic(out/'validation_replay.json',{'max_per_image_dice_error':max_error,'passed':max_error<=1e-4,'checkpoint_sha256':lock['checkpoint_sha256'],'validation':replay})
        assert max_error<=1e-4,'Validation replay differs; test not accessed'
        atomic(dest/'test_access.json',{'created':time.time(),'checkpoint_sha256':lock['checkpoint_sha256'],'validation_replay_passed':True,'selection':'fixed best native-resolution validation mean Dice'})
        result=evaluate(model,data['splits']['test'],dest)
        result.update(dataset='crack500',method=a.stage,seed=a.seed,checkpoint_sha256=lock['checkpoint_sha256'],selected_step=ckpt['step'],protocol=data['protocol'],inference=lock['inference'],test_lock_sha256=sha(out/'test_lock.json'))
        atomic(dest/'summary.json',result); print(json.dumps({'test_completed':True,'mean':result['mean']}),flush=True); return
    out.mkdir(parents=True,exist_ok=False)
    state={'status':'starting','created':time.time(),'pid':os.getpid(),'steps':a.steps,'stage':a.stage}; atomic(out/'status.json',state)
    try:
        cfg=vars(a)|{'dataset':'crack500','manifest_sha256':MANIFEST_SHA,'model':'ConvNeXt-Tiny / independent UNet' if a.stage=='base' else 'Frozen Crack500 ConvNeXt-Tiny/UNet plus v8 unrestricted residual head',
            'pretrained_sha256':WEIGHT_SHA,'training_data':'cleaned frozen243train/49val/200test; existing cache and native GT',
            'crop':'same historical Crack500 max1024 cache,512 crops,half foreground-centered,flips/rot90/brightness',
            'selection':'maximum native-resolution validation mean Dice; no TopoMortar-specific floor or stress score',
            'test':'max1024;512 windows stride384;mean probabilities;resize to original GT;threshold0.5;no TTA/postprocessing',
            'supervision':'SCNP single view' if a.stage=='base' else 'exact v8 SCNP two aligned views plus structure-weighted JS;paired color and amodal perturbation',
            'optimizer':'AdamW encoder3e-5/decoder3e-4 wd0.01' if a.stage=='base' else 'AdamW trainable residual head3e-4 wd0.01;frozen local and DINO',
            'scheduler':'cosine1to0.01','clip_norm':12,'precision':'FP16 model/FP32 supervised and JS losses;GradScaler',
            'paired_js_coefficient':None if a.stage=='base' else '0.5*min(step/600,1)',
            'test_history':'post-development comparison; original Crack500 test baseline already known; new checkpoints selected by validation only',
            'method_transfer':'v8 context architecture and training loss reused; dataset-specific local backbone trained on Crack500 only; no TopoMortar segmentation weights',
            'comparison':'original ResNet SCNP is historical recipe comparator;new ConvNeXt base and matched local refinement are separately retained',
            'test_accessed':False,'external_resource':read(PHASE/'resource_manifest.json')}
        atomic(out/'config.json',cfg); shutil.copy2(manifest,out/'data_manifest.json')
        shutil.copytree(REPO/'research',out/'source_snapshot/research',ignore=shutil.ignore_patterns('__pycache__'))
        for name in ['reference','SCNP']:
            shutil.copytree(REPO/name,out/'source_snapshot'/name,ignore=shutil.ignore_patterns('__pycache__','.git'))
        sources={str(f.relative_to(REPO)):sha(f) for f in sorted((REPO/'research').glob('*.py'))}
        sources.update({str(f.relative_to(REPO)):sha(f) for f in sorted((REPO/'reference').rglob('*.py'))})
        sources.update({str(f.relative_to(REPO)):sha(f) for f in sorted((REPO/'SCNP').rglob('*.py'))})
        atomic(out/'source_hashes.json',sources)
        model=model_for(a.stage,a.warm).cuda()
        frozen_before=verify_frozen(model) if a.stage!='base' else None
        atomic(out/'architecture.json',{'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'frozen_before':frozen_before})
        atomic(out/'initialization.json',{'parameter_sha256':digest(model),'trainable_sha256':trainable_hash(model) if a.stage!='base' else None,'base_checkpoint_sha256':sha(a.warm) if a.warm else None})
        if a.stage=='base':
            enc=list(model.encoder.parameters()); enc_ids={id(p) for p in enc}
            opt=torch.optim.AdamW([{'params':enc,'lr':3e-5},{'params':[p for p in model.parameters() if id(p) not in enc_ids],'lr':3e-4}],weight_decay=.01)
        else: opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.01)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda it:.01+.99*(1+math.cos(math.pi*it/a.steps))/2)
        scaler=torch.amp.GradScaler('cuda'); criterion=load_upstream_loss().cuda()
        g=torch.Generator().manual_seed(a.seed+12345)
        loader=DataLoader(TrainingData(data['splits']['train']),batch_size=a.batch,shuffle=True,drop_last=True,num_workers=a.workers,persistent_workers=a.workers>0,worker_init_fn=seed_worker,generator=g,pin_memory=True)
        val=data['splits']['val'][:2] if a.smoke else data['splits']['val']
        view_rng=torch.Generator(device='cuda').manual_seed(a.seed+67890)
        occ_rng=torch.Generator(device='cuda').manual_seed(a.seed+9876)
        best=-1.; iterator=iter(loader); start=time.time(); torch.cuda.reset_peak_memory_stats()
        observed_grad={}; observed_identity=False
        def checkpoint(step):
            return {'model':model.state_dict(),'step':step,'config':cfg,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),
                'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),
                'loader_generator':g.get_state(),'paired_view_generator':view_rng.get_state(),'occlusion_generator':occ_rng.get_state(),
                'resume_note':'Worker augmentation streams not exactly restorable;no silent resume as an identical run'}
        for step in range(1,a.steps+1):
            try:x,y,geom,ids=next(iterator)
            except StopIteration:iterator=iter(loader);x,y,geom,ids=next(iterator)
            x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);geom=geom.cuda(non_blocking=True)
            model.train();opt.zero_grad(set_to_none=True);diag={}
            if a.stage=='base':
                with torch.autocast('cuda',dtype=torch.float16): z=model(x)
                loss=criterion(z.float(),y)
            else:
                x2=paired_view(x,view_rng);x2,occ=amodal_view(x2,y[:,None],geom,occ_rng,step)
                with torch.autocast('cuda',dtype=torch.float16):z=model(torch.cat((x,x2),0))
                z1,z2=z.float().chunk(2,0);supervised=.5*(criterion(z1,y)+criterion(z2,y))
                coeff=.5*min(step/600.,1.);consistency=structure_js(z1,z2,y[:,None],geom)
                loss=supervised+coeff*consistency
                observed_identity=observed_identity or model.identity_verified
                diag={'supervised':float(supervised.detach()),'consistency':float(consistency.detach()),'coefficient':coeff,'occlusion_fraction':float(occ.mean()),**model.last_diagnostic}
            assert torch.isfinite(loss),'Nonfinite loss'
            scaler.scale(loss).backward();scaler.unscale_(opt)
            if a.stage!='base':
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
                if step<=4 or step%25==0:
                    for name,param in model.head.named_parameters():
                        if param.grad is not None:
                            assert torch.isfinite(param.grad).all(),name
                            observed_grad[name]=max(observed_grad.get(name,0.),float(param.grad.abs().sum()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),12);scaler.step(opt);scaler.update();scheduler.step()
            if step==1 or step%25==0 or step==a.steps:
                row={'step':step,'loss':float(loss.detach()),'lr':[p['lr'] for p in opt.param_groups],'elapsed_sec':time.time()-start,'sample_ids':list(ids),'diag':diag,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                state.update(status='running',step=step,elapsed_sec=row['elapsed_sec']);atomic(out/'status.json',state);print(json.dumps(row),flush=True);gpu_guard()
            if step%a.val_every==0 or step==a.steps:
                result=evaluate(model,val);result['step']=step
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(result)+'\n')
                if result['mean']['dice']>best:
                    best=result['mean']['dice'];torch.save(checkpoint(step),out/'best.pt');atomic(out/'best_validation.json',result)
                torch.save(checkpoint(step),out/'last.pt');print(json.dumps({'step':step,'validation_mean':result['mean']}),flush=True)
        if a.stage!='base':
            frozen_after=verify_frozen(model)
            check={'passed':frozen_after==frozen_before,'before':frozen_before,'after':frozen_after,'identity_verified':observed_identity,'max_observed_head_gradient_l1':observed_grad}
            atomic(out/'frozen_verification.json',check);assert check['passed'] and observed_identity
            assert observed_grad.get('output.weight',0)>0
            if a.stage=='dino_context':assert observed_grad.get('project.0.weight',0)>0
        atomic(out/'test_lock.json',{'created':time.time(),'checkpoint_sha256':sha(out/'best.pt'),'manifest_sha256':MANIFEST_SHA,'source_hashes':sources,'inference':cfg['test'],'selected_step':read(out/'best_validation.json')['step'],'selection':cfg['selection']})
        state.update(status='completed',step=a.steps,elapsed_sec=time.time()-start,best_validation_dice=best);atomic(out/'status.json',state)
        atomic(out/'memory_profile.json',{'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2})
    except BaseException as e:
        state.update(status='failed',error=repr(e));atomic(out/'status.json',state);(out/'traceback.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
