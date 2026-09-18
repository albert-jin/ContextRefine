"""Paired-view finetuning with a shared validation-selected warm start."""
import math
import argparse,ast,hashlib,importlib.util,json,os,random,shutil,subprocess,time,traceback
from pathlib import Path
import numpy as np
from PIL import Image
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import torch,yaml,monai
from torch.nn import functional as F
from monai.networks.nets import DynUNet
from monai.networks import one_hot
from monai.transforms import Compose,NormalizeIntensityd
from monai.data import Dataset,DataLoader,worker_init_fn
from scipy.ndimage import distance_transform_edt,maximum_filter
from skimage.morphology import skeletonize
from losses import load_upstream_loss
from chroma_consistency import stress_views,paired_view,structure_js,freeze_encoder_bn,WARM_SHA,WARM_REL
from context_residual import PretrainedUNet,make_optimizer,setup_pretrained_data,WEIGHT_SHA,trainable_hash,verify_frozen
from train import metrics
from amodal_completion import amodal_view

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
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['local_refine','dino_context'],required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--val-every',type=int,default=300);p.add_argument('--run-id',required=True)
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args();out=REPO/'auto_res_logs/runs'/a.run_id;out.mkdir(parents=True,exist_ok=False)
    state={'status':'starting','pid':os.getpid(),'created':time.time()};atomic(out/'status.json',state)
    try:
        assert os.environ.get('SCNP_GPU_GUARDED')=='1' and torch.cuda.is_available()
        torch.set_num_threads(4);monai.utils.set_determinism(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        tr,va,manifest,reference=setup_pretrained_data(a.seed)
        if a.smoke:va=torch.utils.data.Subset(va,[0,1])
        atomic(out/'data_manifest.json',manifest)
        g=torch.Generator().manual_seed(a.seed+12345)
        loader=DataLoader(tr,batch_size=a.batch,shuffle=True,drop_last=True,num_workers=a.workers,
                          persistent_workers=a.workers>0,worker_init_fn=worker_init_fn,generator=g,pin_memory=True)
        cfg=vars(a)|{'dataset':'topomortar','model':'ImageNet1K ConvNeXt-Tiny encoder + independent UNet decoder','reference_commit':'b1f31f20a41b20aa775c21078b2c432c6dcc26b4',
                      'normalization':'RGB /255 before augmentation; clamp then ImageNet mean/std at end, full512 no classification crop','augmentation':'Official geometry/intensity + RandHue, then a second aligned random color/gamma/saturation/grayscale view; same view stream for both arms',
                      'geometry':'precomputed GT skeleton-width priority warped by nearest interpolation with target',
                      'optimizer':'AdamW encoder1e-5 decoder1e-4 wd.01, reset optimizer after shared warm start','scheduler':'cosine multiplier1 to.01','clip_norm':12,
                      'precision':'FP16 autocast+GradScaler; supervised and consistency losses FP32','deep_supervision':'single full-resolution output; no auxiliary heads',
                      'implementation_revision':'topo_free_context_v8','stochastic_depth_max':.1,'batch_norm_note':'ConvNeXt has no BatchNorm running statistics','paired_views':2,'freeze_encoder_bn_running_stats':True,'candidate':{'regularizer':'class-balanced structure-weighted symmetric Jensen-Shannon divergence','geometry_weight':'1+4*clamp(geometry,0,1)','coefficient':'.5 * min(step/600,1), common to both arms','reference':'Augmentation consistency is established; see Hendrycks et al. AugMix ICLR2020 https://arxiv.org/abs/1912.02781; this is an independent two-view variant, not an exact AugMix reproduction'},'warm_start':{'checkpoint':WARM_REL,'sha256':WARM_SHA,'selected_step':1800,'selection':'v4 fixed clean/photometric validation score','source_total_training_steps':3600,'ancestor_convnext_total_steps':12000,'ancestor_selected_step':7200,'additional_steps':a.steps},'stress_validation':['grayscale','rgb_cycle','warm_shadow'],'test_history':'post-development benchmark, earlier test results informed this research round','deterministic_algorithms':True,'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],'pretrained_sha256':WEIGHT_SHA,
                      'threshold':.5,'selection':'max 0.5 clean Dice + 0.5 mean fixed photometric stress Dice; clean floor 0.936 unless smoke; all checkpoints and fallbacks disclosed','test_accessed':False,'size':512}
        cfg.update(model='Frozen v6 ConvNeXt-Tiny/UNet + unrestricted residual head; candidate adds frozen DINOv2-S/14 context',implementation_revision='topo_free_context_v8',optimizer='AdamW trainable residual head3e-4 wd.01; both backbones frozen',freeze_encoder_bn_running_stats=True)
        cfg['candidate']={'comparison':'v8 unrestricted DINO residual versus already-running v7 bounded DINO residual; same pretrained resources, init and budget; archived/overlapping control, not a repeated seed','head':'1536-to64 context projection; concat pooled32 local features,2 base probabilities; 98-to64 GN/GELU block, zero-init1-channel output; bilinear to512','correction':'delta=raw without confidence gate or tanh cap; add [-delta/2,delta/2] to frozen v6 logits','supervision':'SCNP two views + structure-weighted JS, coefficient .5*min(step/600,1), common local occlusion','pretraining_difference':'same frozen LVD-142M DINO and ImageNet1K local weights as matched v7 DINO reference'}
        cfg['warm_start']={'checkpoint':WARM_REL,'sha256':WARM_SHA,'selected_step':1200,'selection':'v6 validation best; post-development research','additional_steps':a.steps,'frozen':True}
        cfg['matched_control_run']='exp_SCNP_topo_context_residual_v7/auto_res_logs/runs/topo_context_dino_context_s0'
        cfg['motivation']='validation-only diagnostic: 47.3% clean errors unreachable under old frozen-logit confidence cap; this candidate removes that representational constraint'
        cfg['external_resource']=json.loads((ROOT/'auto_res_logs/topo_free_context_v8_20260918/resource_manifest.json').read_text())
        cfg['amodal_diagnostic']='fixed synthetic masks on validation only, NOT used for checkpoint selection'
        atomic(out/'config.json',cfg)
        for source in list((REPO/'research').glob('*.py'))+list((REPO/'reference').rglob('*.py'))+list((REPO/'reference/config').glob('*.yaml')):
            dest=out/'source_snapshot'/source.relative_to(REPO);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dest)
        hashes={str(f.relative_to(REPO)):sha(f) for f in (REPO/'research').glob('*.py')};atomic(out/'source_hashes.json',hashes)
        warm=ROOT/WARM_REL
        assert sha(warm)==WARM_SHA,'Warm-start checkpoint changed'
        model=PretrainedUNet(a.method).cuda()
        frozen_before=verify_frozen(model)
        atomic(out/'architecture.json',{'parameters':sum(p.numel() for p in model.parameters()),'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'frozen_sha256':frozen_before,'extra_dino_features':a.method=='dino_context'})
        atomic(out/'initialization.json',{'parameter_sha256':weights_hash(model),'trainable_sha256':trainable_hash(model),'base_checkpoint_sha256':WARM_SHA,'frozen_sha256':frozen_before})
        (out/'model.txt').write_text(str(model))
        opt=make_optimizer(model)
        view_rng=torch.Generator(device='cuda').manual_seed(a.seed+67890)
        occlusion_rng=torch.Generator(device='cuda').manual_seed(a.seed+9876)
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,lambda it:.01+.99*(1+math.cos(math.pi*it/a.steps))/2)
        scaler=torch.amp.GradScaler('cuda');base,cldice=reference_losses();scnp=load_upstream_loss().cuda()
        best=-1.;best_fallback=-1.;start=time.time();iterator=iter(loader);torch.cuda.reset_peak_memory_stats()
        def checkpoint(step):
            return {'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'step':step,
                    'config':cfg,'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all(),'rng_python':random.getstate(),
                    'rng_numpy':np.random.get_state(),'loader_generator':g.get_state(),'paired_view_generator':view_rng.get_state(),'occlusion_generator':occlusion_rng.get_state(),'resume_note':'Worker augmentation streams are not fully restorable; restarting is not exact continuation'}
        for step in range(1,a.steps+1):
            try:batch=next(iterator)
            except StopIteration:iterator=iter(loader);batch=next(iterator)
            x=batch['image'].cuda(non_blocking=True);y=batch['label'].cuda(non_blocking=True);geom=batch['geom'].cuda(non_blocking=True)
            model.train();freeze_encoder_bn(model);opt.zero_grad(set_to_none=True)
            x2=paired_view(x,view_rng)
            masked,occlusion=amodal_view(x2,y,geom,occlusion_rng,step)
            x2=masked
            with torch.autocast('cuda',dtype=torch.float16):pred=model(torch.cat((x,x2),dim=0))
            z1,z2=pred.float().chunk(2,dim=0)
            head_diags=[{},{}];audit_heads=step==1 or step%25==0 or step==a.steps
            if audit_heads:
                def recorder(head):
                    def note(gradient):
                        head_diags[head].update(backward_observed=True,scaled_gradient_l1=float(gradient.detach().abs().sum()),finite=bool(torch.isfinite(gradient).all()))
                    return note
                z1.register_hook(recorder(0));z2.register_hook(recorder(1))
            loss_z2=z2
            budget_diag={'active':False,'replacement':'unrestricted residual context head',**model.last_diagnostic}
            supervised=.5*(scnp(z1,y[:,0].long())+scnp(loss_z2,y[:,0].long()))
            coefficient=.5*min(step/600.,1.)
            consistency=structure_js(z1,z2,y,geom) if coefficient else z1.new_zeros(())
            loss=supervised+coefficient*consistency
            diags=[{'supervised':float(supervised.detach()),'consistency':float(consistency.detach()),'coefficient':coefficient,'occlusion_applied':True,'visibility_budget':budget_diag,'occlusion_fraction':float(occlusion.mean()),'masked_foreground_fraction':float((occlusion*y).sum()/y.sum().clamp_min(1))}]
            assert torch.isfinite(loss),'nonfinite loss'
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),12)
            if audit_heads:
                assert len(head_diags)==2 and all(d.get('backward_observed') for d in head_diags),'Missing paired-view backward path'
                if step==1:assert all(d['finite'] and d['scaled_gradient_l1']>0 for d in head_diags),'Invalid initial head gradients'
            if audit_heads:
                trainable_grads={n:float(p.grad.abs().sum()) for n,p in model.named_parameters() if p.requires_grad and p.grad is not None}
                assert all(p.grad is None for p in model.parameters() if not p.requires_grad), 'Frozen backbone acquired gradient'
            scaler.step(opt);scaler.update();scheduler.step()
            if step==a.steps//4:atomic(out/'warmup_end_parameters.json',{'step':step,'parameter_sha256':weights_hash(model)})
            if step==1 or step%25==0 or step==a.steps:
                row={'step':step,'loss':float(loss.detach()),'lr':opt.param_groups[0]['lr'],'elapsed_sec':time.time()-start,
                     'sample_ids':list(batch['id']),'grad_diags':diags,'head_grad_diags':head_diags,'scale':scaler.get_scale(),'trainable_gradient_l1':trainable_grads,
                     'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2}
                with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
                state.update(status='running',step=step,elapsed_sec=row['elapsed_sec']);atomic(out/'status.json',state);print(json.dumps(row),flush=True)
                smi=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
                for line in smi.splitlines():
                    uuid,total,used=[s.strip() for s in line.split(',')]
                    if uuid==os.environ['CUDA_VISIBLE_DEVICES'] and int(used)/int(total)>=.8:
                        torch.save(checkpoint(step),out/'memory_stop.pt');raise RuntimeError('GPU reached80%; own task stopped')
            if step%a.val_every==0 or step==a.steps:
                model.eval();rows=[];stress_rows=[];amodal_rows=[]
                with torch.no_grad():
                    for item in va:
                        with torch.autocast('cuda',dtype=torch.float16):z=model(item['image'][None].cuda())
                        prediction=(z.float().softmax(1)[0,1]>=.5).cpu().numpy()
                        rows.append({'id':item['id']}|metrics(prediction,np.asarray(item['label'][0])>0))
                        stress={}
                        for name,sx in stress_views(item['image'][None].cuda()).items():
                            with torch.autocast('cuda',dtype=torch.float16):sz=model(sx)
                            sp=(sz.float().softmax(1)[0,1]>=.5).cpu().numpy()
                            sy=np.asarray(item['label'][0])>0
                            stress[name]=float((2*(sp&sy).sum()+1e-8)/(sp.sum()+sy.sum()+1e-8))
                        stress_rows.append({'id':item['id'],**stress})
                        occ_rng=torch.Generator(device='cuda').manual_seed(34567+int(item['id']))
                        ax,am=amodal_view(item['image'][None].cuda(),torch.as_tensor(item['label'],device='cuda')[None],torch.as_tensor(item['geom'],device='cuda')[None],occ_rng,600,probability=1.)
                        with torch.autocast('cuda',dtype=torch.float16):az=model(ax)
                        ap=(az.float().softmax(1)[0,1]>=.5).cpu().numpy();ay=np.asarray(item['label'][0])>0
                        amodal_rows.append({'id':item['id'],'dice':float((2*(ap&ay).sum()+1e-8)/(ap.sum()+ay.sum()+1e-8)),'mask_fraction':float(am.mean())})
                mean={k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='id'};value={'step':step,'mean':mean,'per_image':rows}
                stress_mean={k:float(np.mean([r[k] for r in stress_rows])) for k in ('grayscale','rgb_cycle','warm_shadow')}
                score=.5*mean['dice']+.5*float(np.mean(list(stress_mean.values())))
                value.update(stress_mean=stress_mean,stress_per_image=stress_rows,selection_score=score,clean_floor=.936,amodal_diagnostic={'mean_dice':float(np.mean([r['dice'] for r in amodal_rows])),'per_image':amodal_rows,'used_for_selection':False})
                with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(value)+'\n')
                if score>best_fallback:
                    best_fallback=score;torch.save(checkpoint(step),out/'fallback.pt');atomic(out/'fallback_validation.json',value)
                if (mean['dice']>=.936 or a.smoke) and score>best:
                    best=score;torch.save(checkpoint(step),out/'best.pt');atomic(out/'best_validation.json',value)
                torch.save(checkpoint(step),out/'last.pt');print(json.dumps({'step':step,'validation':mean}),flush=True)
        frozen_after=verify_frozen(model);assert frozen_after==frozen_before, 'Frozen model changed'
        atomic(out/'frozen_verification.json',{'passed':True,'before':frozen_before,'after':frozen_after,'initial_predictions_identical':model.identity_verified})
        atomic(out/'memory_profile.json',{'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2})
        state.update(status='completed',step=a.steps,best_selection_score=best,qualified_checkpoint=(out/'best.pt').exists(),elapsed_sec=time.time()-start);atomic(out/'status.json',state)
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));atomic(out/'status.json',state);(out/'traceback.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
