"""Canonical raw image splits, checksums and matched training cache; no test scoring."""
import argparse,collections,concurrent.futures,hashlib,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt,maximum_filter
from skimage.morphology import skeletonize
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def convert(record):
    image=Image.open(ROOT/record['image']).convert('RGB');mask=Image.open(ROOT/record['mask']).convert('L');record['source_image_size']=list(image.size)
    if record.get('image_rotation_ccw'):
        assert record['split']=='train' and record['image_rotation_ccw']==90
        image=image.transpose(Image.Transpose.ROTATE_90)
    assert image.size==mask.size,record['id']
    y=np.asarray(mask);threshold=record['mask_threshold'];target=y>threshold
    record.update(width=image.width,height=image.height,foreground_fraction=float(target.mean()),image_sha256=sha(ROOT/record['image']),mask_sha256=sha(ROOT/record['mask']))
    if record['split']!='test':
        w,h=image.size;s=min(1.,1024/max(w,h));size=(max(1,round(w*s)),max(1,round(h*s)))
        xx=np.asarray(image.resize(size,Image.Resampling.BILINEAR));yy=np.asarray(Image.fromarray(target.astype('uint8')).resize(size,Image.Resampling.NEAREST)).copy()
        geom=np.zeros(yy.shape,np.float32)
        for region in (yy>0,yy==0):geom+=maximum_filter(skeletonize(region).astype(np.float32)/(1+distance_transform_edt(region)),size=3)*region
        geom[[0,-1],:]=0;geom[:,[0,-1]]=0
        dest=ROOT/record['cache'];dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,image=xx,mask=yy,geometry=geom.astype('float16'));record['cache_sha256']=sha(dest)
    return record
def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=['crack500','deepglobe','fives']);a=p.parse_args();out=ROOT/'data/icassp'/a.dataset;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'manifest.json').exists()
    raw=ROOT/'data/icassp_downloads'/a.dataset/'raw';sources=json.loads((raw.parent/'source_manifest.json').read_text());assert sources['status']=='completed'
    splits={};split_filter=None;orientation_fix={};orientation_review=None;grouped_split=None
    if a.dataset=='crack500':
        for split,folder,count in [('train','traindata',250),('val','valdata',50),('test','testdata',200)]:
            dirs=[d for d in raw.rglob(folder) if any(x.suffix.lower()=='.jpg' for x in d.iterdir() if x.is_file())];assert len(dirs)==1,dirs
            pairs=[(x,x.with_name(x.stem+'_mask.png')) for x in sorted(dirs[0].iterdir()) if x.suffix.lower()=='.jpg'];assert len(pairs)==count;(splits.setdefault(split,[])).extend(pairs)
        filter_path=ROOT/'auto_res_logs/icassp/crack500_split_exclusions.json';split_filter=json.loads(filter_path.read_text());assert split_filter['status']=='frozen_before_training'
        for split,excluded in split_filter['exclude_ids'].items():
            assert set(excluded)<={x.stem for x,_ in splits[split]}
            splits[split]=[(x,y) for x,y in splits[split] if x.stem not in excluded]
        assert {s:len(v) for s,v in splits.items()}==split_filter['clean_counts']
        split_filter={'manifest_path':str(filter_path.relative_to(ROOT)),'manifest_sha256':sha(filter_path),'decision':split_filter}
        orientation_path=ROOT/'auto_res_logs/icassp/crack500_orientation_review.json';orientation_review=json.loads(orientation_path.read_text());orientation_fix=orientation_review['training_image_ccw_degrees'];assert len(orientation_fix)==3
        protocol='Crack500 cleaned capture-family split243/49/200 from original250/50/200 raw images. Before model fitting, seven training and one validation images overlapping higher-priority splits were excluded after rotation-aligned identity review. All200 test images and original files unchanged. Explicit exclusions and evidence frozen in split_filter. All methods share this protocol; not directly comparable to published original-split scores. Crop directories and author result maps excluded. Train cache max side1024; final metrics on native original GT; ResNet34 UNet, not exact SCNP DeepLab implementation.'
    elif a.dataset=='fives':
        originals=[(x,raw/'train/Ground truth'/x.name) for x in sorted((raw/'train/Original').glob('*.png'))];test=[(x,raw/'test/Ground truth'/x.name) for x in sorted((raw/'test/Original').glob('*.png'))]
        assert len(originals)==600 and len(test)==200
        grouped_path=ROOT/'auto_res_logs/icassp/fives_grouped_split.json';grouped_split=json.loads(grouped_path.read_text());assert grouped_split['status']=='frozen_before_training'
        splits={s:[(ROOT/r['image'],ROOT/r['mask']) for r in rr] for s,rr in grouped_split['split_records'].items()}
        assert {p for pairs in splits.values() for p in pairs}==set(originals+test)
        assert {s:len(v) for s,v in splits.items()}==grouped_split['counts']
        assert len(splits['train'])==480 and len(splits['val'])==120
        protocol='Original FIVES600/200 training/test records retained; disease-stratified, source-image-SHA-grouped fixed seed20260917 validation120 from training600 gives480/120/200 as SCNP TableG.4 counts. Five duplicate-image pairs inside source training and one inside source test have different masks; all annotations retained in their assigned split. Test has200records/199unique input images. Frozen explicit grouped split prevents exact-image train/val leakage. Exact author validation IDs not available; patient separation unverified. Train preprocessing max1024, final evaluation native2048 GT. ResNet34 UNet, not original nnUNet; not directly comparable to original nnUNet scores.'
    else:
        pairs=[(x,x.with_name(x.stem.replace('_sat','_mask')+'.png')) for x in sorted((raw/'train').glob('*_sat.jpg'))];assert len(pairs)==6226
        random.Random(20260917).shuffle(pairs);splits={'train':pairs[:4357],'val':pairs[4357:4980],'test':pairs[4980:]}
        protocol='Fixed seed20260917 split of the 6226 public labeled training images into4357/623/1246; same counts but author SCNP IDs unavailable, so not claimed identical split. Official unlabeled validation/test not used. Native GT threshold128.'
    records=[]
    for split,pairs in splits.items():
        for image,mask in pairs:
            assert mask.exists();arr=np.asarray(Image.open(mask).convert('L'));threshold=127 if arr.max()>1 else 0
            ident=image.parent.parent.name+'_'+image.stem if a.dataset=='fives' else image.stem
            cache_dir='cache_grouped_v1' if a.dataset=='fives' else 'cache'
            records.append({'id':ident,'split':split,'image':str(image.relative_to(ROOT)),'mask':str(mask.relative_to(ROOT)),'mask_threshold':threshold,'cache':str((out/cache_dir/split/(ident+'.npz')).relative_to(ROOT))})
            if split=='train' and ident in orientation_fix:records[-1]['image_rotation_ccw']=orientation_fix[ident]
    assert len({r['id'] for r in records})==len(records),'Cross-split duplicate image ID'
    processed=[];started=time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=6) as pool:
        for i,r in enumerate(pool.map(convert,records)):
            processed.append(r)
            if i%25==0:(out/'preparation_progress.json').write_text(json.dumps({'processed':i+1,'total':len(records),'elapsed_sec':time.time()-started}))
    sets={s:{r['image_sha256'] for r in processed if r['split']==s} for s in splits}
    assert not (sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test']),'Cross-split exact image duplicate'
    assert np.mean([r['foreground_fraction'] for r in processed if r['split']=='train'])>.0001,'Degenerate labels'
    result={'dataset':a.dataset,'status':'ready','created':time.time(),'source':sources,'protocol':protocol,'counts':{s:len(v) for s,v in splits.items()},'splits':{s:[r for r in processed if r['split']==s] for s in splits},'audit':{'cross_split_ids_disjoint':True,'cross_split_image_hashes_disjoint':True,'image_mask_dimensions_match':True,'train_foreground_mean':float(np.mean([r['foreground_fraction'] for r in processed if r['split']=='train'])),'test_scores_accessed':False}}
    if split_filter:result['split_filter']=split_filter
    if orientation_review:result['training_orientation_review']=orientation_review;result['protocol']+=' Three training-only source images rotated CCW90 in derived cache to align unchanged native masks, with reviewed overlay evidence; raw source files unchanged.'
    if grouped_split:result['grouped_split']={'path':str(grouped_path.relative_to(ROOT)),'sha256':sha(grouped_path),'policy':grouped_split['policy'],'unique_images_per_split':grouped_split['unique_images_per_split']}
    (out/'manifest.json').write_text(json.dumps(result,indent=2));print(json.dumps({'dataset':a.dataset,'counts':result['counts'],'audit':result['audit']}))
if __name__=='__main__':main()
