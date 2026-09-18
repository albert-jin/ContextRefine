"""Locked saved ResNet evaluation; validation replay must pass before test use."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
from PIL import Image
import torch,monai
from test_metrics import metrics
ROOT=Path(__file__).resolve().parents[1];LOCK=ROOT/'auto_res_logs/topo_free_context_v8_20260918/test_lock.json'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def atomic(p,d):
    t=p.with_suffix('.tmp');t.write_text(json.dumps(d,indent=2));t.replace(p)
def verify(plan):
    for j in plan['jobs']:
        assert sha(ROOT/j['checkpoint'])==j['checkpoint_sha256']
        for group in (j['sources'],j['records']):
            for p,h in group.items():assert sha(ROOT/p)==h,p
    assert sha(ROOT/plan['public_reference_path'])==plan['public_reference_sha256']
    assert sha(ROOT/'scripts/test_metrics.py')==plan['metric_code_sha256']
    assert sha(ROOT/'auto_res_logs/results/test_metric_verification.json')==plan['metric_verification_sha256']
def gpu_check():
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    for line in raw.splitlines():
        uuid,total,used=[v.strip() for v in line.split(',')]
        if uuid==os.environ['CUDA_VISIBLE_DEVICES']:assert int(used)/int(total)<.8,'GPU80% guard'
def main():
    p=argparse.ArgumentParser();p.add_argument('--id',required=True);p.add_argument('--validation',action='store_true');a=p.parse_args()
    assert os.environ.get('SCNP_GPU_GUARDED')=='1' and torch.cuda.is_available()
    plan=read(LOCK);verify(plan);job=next(x for x in plan['jobs'] if x['id']==a.id)
    repo=ROOT/job['repo'];source=repo/'auto_res_logs/runs'/job['run_id'];mode='validation' if a.validation else 'test'
    out=ROOT/'auto_res_logs/topo_free_context_v8_20260918/evaluation'/job['id']/mode;out.mkdir(parents=True,exist_ok=False)
    state={'status':'starting','pid':os.getpid(),'created':time.time(),'mode':mode};atomic(out/'status.json',state)
    try:
        if a.validation:
            rows=[r|{'image':f"data/TopoMortar/dataset/val/images/{r['id']}.png",'mask':f"data/TopoMortar/dataset/val/accurate/{r['id']}.png",'group':'validation','partition':'validation'} for r in read(source/'data_manifest.json')['val']]
            manifest_hash=sha(source/'data_manifest.json')
        else:
            for j in plan['jobs']:
                v=read(ROOT/'auto_res_logs/topo_free_context_v8_20260918/evaluation'/j['id']/'validation/summary.json');assert v['validation_equivalence']['passed']
            manifest=ROOT/'data/TopoMortar/test_download_manifest.json';rows=read(manifest)['images'];manifest_hash=sha(manifest)
            assert len(rows)==350 and {int(r['id']) for r in rows}==set(range(71,421))
        torch.set_num_threads(4);monai.utils.set_determinism(0);torch.use_deterministic_algorithms(job['deterministic_algorithms'])
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;gpu_check()
        sys.path.insert(0,str(repo/'research'))
        from context_residual import PretrainedUNet,ImageNetNormalize,setup_pretrained_data
        saved=torch.load(ROOT/job['checkpoint'],map_location='cpu',weights_only=False)
        assert saved['step']==job['step'];model=PretrainedUNet(saved['config']['method']).cuda();model.load_state_dict(saved['model'],strict=True);model.eval();del saved
        normalizer=ImageNetNormalize(from_bytes=True)
        if a.validation:
            _,va,_,_=setup_pretrained_data(0);reference={r['id']:r for r in va}
        per_image=[];predictions={};equivalence=[];torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            for index,row in enumerate(rows):
                xp=ROOT/row['image'];yp=ROOT/row['mask'];assert sha(xp)==row['image_sha256'] and sha(yp)==row['mask_sha256']
                raw=np.asarray(Image.open(xp).convert('RGB'),dtype=np.float32).transpose(2,0,1).copy();y=np.asarray(Image.open(yp))>0
                assert raw.shape==(3,512,512) and y.shape==(512,512)
                x=normalizer({'image':torch.from_numpy(raw)})['image'][None].cuda()
                with torch.autocast('cuda',dtype=torch.float16):z=model(x)
                pred=(z.float().softmax(1)[0,1]>=.5).cpu().numpy()
                if a.validation:
                    ref=reference[row['id']];refx=ref['image'][None].cuda()
                    with torch.autocast('cuda',dtype=torch.float16):rz=model(refx)
                    rp=(rz.float().softmax(1)[0,1]>=.5).cpu().numpy()
                    equivalence.append({'id':row['id'],'input_identical':bool(torch.equal(x,refx)),'prediction_identical':bool(np.array_equal(pred,rp)),'target_identical':bool(np.array_equal(y,np.asarray(ref['label'][0])>0))})
                per_image.append({'id':row['id'],'partition':row['partition'],'group':row['group']}|metrics(pred,y))
                predictions[row['id']]=np.packbits(pred.reshape(-1))
                if index%25==0 or index+1==len(rows):
                    gpu_check();state.update(status='running',images=index+1,total=len(rows),elapsed_sec=time.time()-state['created']);atomic(out/'status.json',state)
        groups={}
        for name in sorted({r['group'] for r in rows}|{r['partition'] for r in rows}|{'overall'}):
            selected=[r for r in per_image if name=='overall' or r['group']==name or r['partition']==name]
            keys=[k for k in selected[0] if k not in ('id','partition','group','author_cldice_defined')]
            groups[name]={'images':len(selected),'mean':{k:float(np.mean([r[k] for r in selected])) for k in keys},
                          'author_cldice_undefined_images':sum(not r['author_cldice_defined'] for r in selected)}
        result={'id':job['id'],'run_id':job['run_id'],'role':job['role'],'checkpoint':job['checkpoint'],'checkpoint_sha256':job['checkpoint_sha256'],'step':job['step'],
                'method':job['method'],'seed':job['seed'],'mode':mode,'test_accessed':not a.validation,'single_seed':True,
                'test_history':plan['test_history'],'lock_sha256':sha(LOCK),'data_manifest_sha256':manifest_hash,'inference':plan['inference'],'deterministic_algorithms':job['deterministic_algorithms'],
                'sources':{p.name:sha(p) for p in (Path(__file__),ROOT/'scripts/test_metrics.py')},'groups':groups,'per_image':per_image,
                'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2,'elapsed_sec':time.time()-state['created'],
                'topology_note':'Foreground8/background4. Author empty-foreground beta0 convention and undefined clDice counts disclosed separately.'}
        if a.validation:
            historic={r['id']:r for r in read(source/'best_validation.json')['per_image']}
            errors={k:max(abs(r[k]-historic[r['id']][k]) for r in per_image) for k in ('dice','foreground_iou','cldice','betti0_error')}
            passed=all(all(r[k] for k in ('input_identical','prediction_identical','target_identical')) for r in equivalence) and errors['dice']<plan['validation_guard']['historical_max_per_image_dice_error']
            result['validation_equivalence']={'passed':passed,'historical_max_per_image_error':errors,'direct_paths':equivalence,'guard':plan['validation_guard']}
            atomic(out/'verification_before_assert.json',result);assert passed,result['validation_equivalence']
        np.savez_compressed(out/'predictions_packed.npz',**predictions);result['predictions_sha256']=sha(out/'predictions_packed.npz');atomic(out/'summary.json',result)
        state.update(status='completed',elapsed_sec=result['elapsed_sec']);atomic(out/'status.json',state)
        print(json.dumps({k:v for k,v in result.items() if k!='per_image'},indent=2))
    except BaseException as exc:
        state.update(status='failed',error=repr(exc));atomic(out/'status.json',state);(out/'traceback.txt').write_text(traceback.format_exc());raise
if __name__=='__main__':main()
