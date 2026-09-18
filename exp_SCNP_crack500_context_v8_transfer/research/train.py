"""Matched V2 research runner. All candidates/controls share a linear two-class head."""
import argparse,hashlib,json,os,random,shutil,subprocess,sys,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt,maximum_filter,label
from skimage.morphology import skeletonize
import torch
from torch.nn import functional as F
from model import FeatureUNet
from losses import softmax_cedice,load_upstream_loss
from innovations import auxiliary
from mass_budget import allocate_logits
REPO=Path(__file__).resolve().parents[1];ROOT=REPO.parent

def atomic(path,data):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(data,indent=2));tmp.replace(path)

def load_split(dataset,split,size):
    if dataset=='topomortar':base=ROOT/'data/TopoMortar/dataset'/split;expected={'train':50,'val':20}[split]
    elif dataset=='chase':base=ROOT/'data/CHASE'/split;expected={'train':16,'val':4}[split]
    elif dataset=='deepcrack':base=ROOT/'data/DeepCrack_grouped'/split;expected=json.loads((base.parent/'splits.json').read_text())['split_counts'][split]
    else:base=ROOT/'data/CrackForest'/split;expected={'train':82,'val':18}[split]
    xs=[];ys=[];priorities=[];manifest=[]
    images=sorted((base/'images').glob('*.png'),key=lambda p:p.stem);assert len(images)==expected
    for path in images:
        mp=base/'accurate'/path.name;raw=np.asarray(Image.open(mp));assert set(np.unique(raw))<={0,1,255}
        image=np.asarray(Image.open(path).convert('RGB').resize((size,size),Image.Resampling.BILINEAR),dtype=np.float32)/255
        y=np.asarray(Image.fromarray((raw>0).astype('uint8')).resize((size,size),Image.Resampling.NEAREST)).copy()
        image=(image-np.array([.485,.456,.406],dtype=np.float32))/np.array([.229,.224,.225],dtype=np.float32)
        priority=np.zeros(y.shape,np.float32)
        for region in (y>0,y==0):
            sk=skeletonize(region);d=distance_transform_edt(region)
            priority+=maximum_filter(sk.astype(np.float32)/(1+d),size=3)*region
        priority[[0,-1],:]=0;priority[:,[0,-1]]=0
        xs.append(torch.from_numpy(image.transpose(2,0,1).copy()));ys.append(torch.from_numpy(y.astype('int64')))
        priorities.append(torch.from_numpy(priority.astype('float32')))
        manifest.append({'id':path.stem,'image':str(path.relative_to(ROOT)),'mask':str(mp.relative_to(ROOT)),
                         'image_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'mask_sha256':hashlib.sha256(mp.read_bytes()).hexdigest()})
    if dataset=='cfd':
        plan=json.loads((ROOT/'data/CrackForest/splits.json').read_text())
        assert plan['preparation_complete'] and plan['label_audit_version']==3
        assert [r['id'] for r in manifest]==plan['split_ids'][split]
        byid={r['id']:r for r in plan['files'][split]}
        assert all(r['image_sha256']==byid[r['id']]['image_sha256'] and r['mask_sha256']==byid[r['id']]['label_sha256'] for r in manifest)
    if dataset=='deepcrack':
        plan=json.loads((ROOT/'data/DeepCrack_grouped/splits.json').read_text());assert plan['preparation_complete'] and plan['grouped_version']==1
        assert not ({s.split('-')[0] for s in plan['split_ids']['train']}&{s.split('-')[0] for s in plan['split_ids']['val']})
        assert [r['id'] for r in manifest]==plan['split_ids'][split]
        byid={r['id']:r for r in plan['files'][split]}
        assert all(r['image_sha256']==byid[r['id']]['image_sha256'] and r['mask_sha256']==byid[r['id']]['label_sha256'] for r in manifest)
    if dataset=='chase':
        plan=json.loads((ROOT/'data/CHASE/splits.json').read_text());assert plan['preparation_complete']
        assert [r['id'] for r in manifest]==plan['split_ids'][split]
        byid={r['id']:r for r in plan['files'][split]}
        assert all(r['image_sha256']==byid[r['id']]['image_sha256'] and r['mask_sha256']==byid[r['id']]['label_sha256'] for r in manifest)
        assert not (set(plan['subjects']['train'])&set(plan['subjects']['val']))
    return torch.stack(xs),torch.stack(ys),torch.stack(priorities),manifest

def metrics(pred,target):
    pred=pred.astype(bool);target=target.astype(bool);inter=(pred&target).sum()
    dice=(2*inter+1e-8)/(pred.sum()+target.sum()+1e-8)
    iou=(inter+1e-8)/((pred|target).sum()+1e-8)
    bg=((~pred)&(~target)).sum()/max(1,((~pred)|(~target)).sum())
    ps=skeletonize(pred);ts=skeletonize(target)
    precision=(ps&target).sum()/max(1,ps.sum());recall=(ts&pred).sum()/max(1,ts.sum())
    cl=2*precision*recall/max(1e-8,precision+recall)
    if not pred.any() and not target.any():cl=1.
    return {'dice':float(dice),'foreground_iou':float(iou),'miou':float((iou+bg)/2),'cldice':float(cl),
            'betti0_error':int(abs(label(pred,np.ones((3,3)))[1]-label(target,np.ones((3,3)))[1])),
            'predicted_fraction':float(pred.mean()),'target_fraction':float(target.mean())}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--method',choices=['base','scnp','margin_joint','balanced_joint','uniform','uncertainty','geometry','softtail','path','path_unbalanced','cldice','mass_uniform','mass_error','mass_geometry','mass_completion'],required=True)
    p.add_argument('--dataset',choices=['topomortar','cfd','deepcrack','chase'],default='deepcrack')
    p.add_argument('--steps',type=int,default=1500);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--size',type=int,default=512);p.add_argument('--batch',type=int,default=2)
    p.add_argument('--lr',type=float,default=.001);p.add_argument('--weight-decay',type=float,default=.1)
    p.add_argument('--lam',type=float,default=1.);p.add_argument('--val-every',type=int,default=300)
    p.add_argument('--run-id',required=True)
    a=p.parse_args();out=REPO/'auto_res_logs/runs'/a.run_id;out.mkdir(parents=True,exist_ok=False)
    state={'run_id':a.run_id,'status':'starting','pid':os.getpid(),'created':time.time()};atomic(out/'status.json',state)
    try:
        assert os.environ.get('SCNP_GPU_GUARDED')=='1' and torch.cuda.is_available()
        torch.set_num_threads(4);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        x,y,geom,tm=load_split(a.dataset,'train',a.size);vx,vy,_,vm=load_split(a.dataset,'val',a.size)
        assert not ({r['image_sha256'] for r in tm}&{r['image_sha256'] for r in vm})
        data={'train':tm,'val':vm};atomic(out/'data_manifest.json',data)
        cfg=vars(a)|{'model':'MONAI UNet feature16 + unrestricted linear 16-to-2 head','protocol':'phase4_forward_preserving_gradients',
                     'torch':torch.__version__,'initialization':'random','test_accessed':False,'threshold':.5,
                     'gradient_budget_settings':{'strength':8,'warmup_fraction':0.25,'ramp_fraction':0.25,'normalization':'per-image/channel/true-label logit gradient L1'},'augmentation':'paired hflip/vflip/rot90','normalization':'ImageNet RGB mean/std','device':os.environ.get('CUDA_VISIBLE_DEVICES'),
                     'selection':'best mean validation Dice; fixed final also reported','source_commit':'1649403189f51c97db8629dbdf1984c184646270'}
        atomic(out/'config.json',cfg)
        if a.dataset=='chase':shutil.copyfile(ROOT/'data/CHASE/splits.json',out/'dataset_protocol.json')
        if a.dataset=='deepcrack':shutil.copyfile(ROOT/'data/DeepCrack_grouped/splits.json',out/'dataset_protocol.json')
        sources=list(Path(__file__).parent.glob('*.py'))+[REPO/'SCNP/experiments/Detectron2/loss.py']
        hashes={str(f.relative_to(REPO)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sources};atomic(out/'source_hashes.json',hashes)
        for f in sources:
            q=out/'source_snapshot'/f.relative_to(REPO);q.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(f,q)
        model=FeatureUNet().cuda();(out/'model.txt').write_text(str(model))
        init=hashlib.sha256()
        for key,value in model.state_dict().items():init.update(key.encode());init.update(value.detach().cpu().numpy().tobytes())
        atomic(out/'initialization.json',{'parameter_sha256':init.hexdigest(),'parameters':sum(p.numel() for p in model.parameters())})
        optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=a.weight_decay)
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,a.steps)
        upstream=load_upstream_loss().cuda();sampler=torch.Generator().manual_seed(a.seed+12345)
        best=-1;started=time.time();torch.cuda.reset_peak_memory_stats()
        def checkpoint(step):
            return {'model':model.state_dict(),'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'step':step,
                    'config':cfg,'rng_sampler':sampler.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),
                    'rng_python':random.getstate(),'rng_numpy':np.random.get_state()}
        for step in range(1,a.steps+1):
            model.train();ix=torch.randint(len(x),(a.batch,),generator=sampler)
            xx=x[ix].clone();yy=y[ix].clone();gg=geom[ix].clone()
            if torch.rand((),generator=sampler)<.5:xx=xx.flip(-1);yy=yy.flip(-1);gg=gg.flip(-1)
            if torch.rand((),generator=sampler)<.5:xx=xx.flip(-2);yy=yy.flip(-2);gg=gg.flip(-2)
            k=int(torch.randint(4,(),generator=sampler));xx=xx.rot90(k,(-2,-1));yy=yy.rot90(k,(-2,-1));gg=gg.rot90(k,(-2,-1))
            xx=xx.cuda();yy=yy.cuda();gg=gg[:,None].cuda();optimizer.zero_grad(set_to_none=True)
            logits=model(xx);base=softmax_cedice(logits,yy);diag={};aux=torch.zeros((),device='cuda')
            if a.method=='base':loss=base
            elif a.method=='scnp':loss=upstream(logits,yy)
            elif a.method.startswith('mass_'):
                allocated,diag=allocate_logits(logits,yy,gg,a.method,step,a.steps);loss=upstream(allocated,yy)
            else:
                aux,diag=auxiliary(a.method,logits[:,1:2]-logits[:,:1],yy[:,None].float(),gg)
                loss=base+a.lam*aux
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
            loss.backward();optimizer.step();scheduler.step()
            if step==a.steps//4:
                hh=hashlib.sha256()
                for key,value in model.state_dict().items():hh.update(key.encode());hh.update(value.detach().cpu().numpy().tobytes())
                atomic(out/'warmup_end_parameters.json',{'step':step,'parameter_sha256':hh.hexdigest()})
            if step==1 or step%50==0:
                row={'step':step,'loss':float(loss.detach()),'base':float(base.detach()),'aux':float(aux.detach()),'diag':diag,
                     'lr':optimizer.param_groups[0]['lr'],'elapsed_sec':time.time()-started,'sample_indices':ix.tolist(),
                     'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                print(json.dumps(row),flush=True)
                state.update(status='running',step=step,elapsed_sec=row['elapsed_sec']);atomic(out/'status.json',state)
                snapshot=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
                for line in snapshot.splitlines():
                    uuid,total,used=[s.strip() for s in line.split(',')]
                    if uuid==os.environ.get('CUDA_VISIBLE_DEVICES') and int(used)/int(total)>=.8:
                        torch.save(checkpoint(step),out/'memory_stop.pt');raise RuntimeError('GPU usage reached80%; saved own task')
            if step%a.val_every==0 or step==a.steps:
                model.eval();rows=[]
                with torch.no_grad():
                    for i in range(len(vx)):
                        prob=model(vx[i:i+1].cuda()).softmax(1)[0,1].cpu().numpy();pred=prob>=.5
                        rows.append({'id':vm[i]['id']}|metrics(pred,vy[i].numpy())|{'max_probability':float(prob.max())})
                mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='id'}
                val={'step':step,'mean':mean,'per_image':rows}
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(val)+'\n')
                if mean['dice']>best:
                    best=mean['dice'];torch.save(checkpoint(step),out/'best.pt');atomic(out/'best_validation.json',val)
                torch.save(checkpoint(step),out/'last.pt');print(json.dumps({'step':step,'validation':mean}),flush=True)
        state.update(status='completed',step=a.steps,elapsed_sec=time.time()-started,best_dice=best);atomic(out/'status.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));atomic(out/'status.json',state);(out/'traceback.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
