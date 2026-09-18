"""Matched native-label main experiments; fixed preprocessing and validation-only selection."""
import argparse,hashlib,json,math,os,random,shutil,sys,time,traceback
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader
ROOT=Path(__file__).resolve().parents[2];REPO=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from test_metrics import metrics
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def atomic(p,d):p=Path(p);q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def norm(x):return (x/255.-x.new_tensor([.485,.456,.406])[:,None,None])/x.new_tensor([.229,.224,.225])[:,None,None]
def resize_image(image,longside=1024):
    w,h=image.size;s=min(1.,longside/max(w,h));return image.resize((max(1,round(w*s)),max(1,round(h*s))),Image.Resampling.BILINEAR)
def gpu_guard():
    import subprocess,csv
    raw=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
    for row in csv.reader(raw.splitlines()):
        if row[0].strip()==os.environ['CUDA_VISIBLE_DEVICES']:assert int(row[2])/int(row[1])<.8,'GPU80% protection'
class TrainingData(Dataset):
    def __init__(self,rows):self.rows=rows
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i];data=np.load(ROOT/row['cache']);x=torch.from_numpy(data['image'].copy()).permute(2,0,1).float();y=torch.from_numpy(data['mask'].astype('int64'));g=torch.from_numpy(data['geometry'].astype('float32'))
        h,w=y.shape;ph=max(0,512-h);pw=max(0,512-w)
        if ph or pw:x=F.pad(x,(0,pw,0,ph),mode='replicate');y=F.pad(y,(0,pw,0,ph));g=F.pad(g,(0,pw,0,ph))
        h,w=y.shape
        if random.random()<.5 and y.any():
            pos=torch.nonzero(y);cy,cx=pos[random.randrange(len(pos))].tolist();top=max(0,min(h-512,cy-random.randrange(512)));left=max(0,min(w-512,cx-random.randrange(512)))
        else:top=random.randrange(h-512+1);left=random.randrange(w-512+1)
        x=x[:,top:top+512,left:left+512];y=y[top:top+512,left:left+512];g=g[top:top+512,left:left+512]
        for dim in (-1,-2):
            if random.random()<.5:x=x.flip(dim);y=y.flip(dim);g=g.flip(dim)
        k=random.randrange(4);x=x.rot90(k,(-2,-1));y=y.rot90(k,(-2,-1));g=g.rot90(k,(-2,-1))
        x=(x*(.85+.3*random.random())+(-.05+.1*random.random())*255).clamp(0,255)
        return norm(x),y,g[None],row['id']
def seed_worker(worker_id):
    s=torch.initial_seed()%2**32;random.seed(s);np.random.seed(s)
@torch.no_grad()
def predict(model,image):
    raw=resize_image(image);x=norm(torch.from_numpy(np.asarray(raw).copy()).permute(2,0,1).float());h,w=x.shape[-2:]
    x=F.pad(x,(0,max(0,512-w),0,max(0,512-h)),mode='replicate');hh,ww=x.shape[-2:]
    starts=lambda n:sorted(set(list(range(0,max(1,n-512+1),384))+[n-512]))
    total=torch.zeros((hh,ww),device='cuda');counts=torch.zeros_like(total)
    for top in starts(hh):
        for left in starts(ww):
            with torch.autocast('cuda',dtype=torch.float16):z=model(x[:,top:top+512,left:left+512][None].cuda())
            total[top:top+512,left:left+512]+=z.float().softmax(1)[0,1];counts[top:top+512,left:left+512]+=1
    prob=(total/counts)[:h,:w]
    return F.interpolate(prob[None,None],size=(image.height,image.width),mode='bilinear',align_corners=False)[0,0].cpu().numpy()
def evaluate(model,rows,out=None):
    model.eval();values=[];predictions={};shapes={};start=time.time()
    for i,r in enumerate(rows):
        if out:assert sha(ROOT/r['image'])==r['image_sha256'] and sha(ROOT/r['mask'])==r['mask_sha256']
        image=Image.open(ROOT/r['image']).convert('RGB');target=np.asarray(Image.open(ROOT/r['mask']).convert('L'))>r['mask_threshold']
        assert target.shape==(image.height,image.width)
        prob=predict(model,image);pred=prob>=.5;values.append({'id':r['id'],**metrics(pred,target)})
        if out:
            predictions[r['id']]=np.packbits(pred.reshape(-1));shapes[r['id']]=list(pred.shape)
            if i%10==0:atomic(out/'progress.json',{'images':i+1,'total':len(rows),'elapsed_sec':time.time()-start})
        if i%10==0:gpu_guard()
    keys=[k for k,v in values[0].items() if k!='id' and type(v) in (float,int)]
    result={'images':len(rows),'mean':{k:float(np.mean([r[k] for r in values])) for k in keys},'per_image':values,'elapsed_sec':time.time()-start}
    if out:np.savez_compressed(out/'predictions_packed.npz',**predictions);atomic(out/'prediction_shapes.json',shapes)
    return result
