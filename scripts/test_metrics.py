"""Independent test metrics with explicit foreground connectivity conventions."""
import numpy as np
from scipy.ndimage import label
from skimage.morphology import skeletonize
def betti(mask):
    mask=np.asarray(mask,dtype=bool)
    b0=label(mask,np.ones((3,3),int))[1]
    # Complement of closed pixel squares is four-connected; discard exterior.
    _,n=label(np.pad(~mask,1,constant_values=True),np.array([[0,1,0],[1,1,1],[0,1,0]]))
    return int(b0),int(n-1)
def metrics(pred,target):
    p=np.asarray(pred,dtype=bool);y=np.asarray(target,dtype=bool);inter=np.logical_and(p,y).sum()
    total=p.sum()+y.sum();union=np.logical_or(p,y).sum()
    dice=float(2*inter/total) if total else 1.
    iou=float(inter/union) if union else 1.
    ps=skeletonize(p);ys=skeletonize(y)
    precision=float((ps&y).sum()/max(1,ps.sum()));recall=float((ys&p).sum()/max(1,ys.sum()))
    cldice=2*precision*recall/(precision+recall) if precision+recall else float(not p.any() and not y.any())
    pb=betti(p);yb=betti(y)
    return {'dice':dice,'foreground_iou':iou,'cldice':cldice,'betti0_error':abs(pb[0]-yb[0]),'betti1_error':abs(pb[1]-yb[1]),
            'pred_betti0':pb[0],'true_betti0':yb[0],'pred_betti1':pb[1],'true_betti1':yb[1],
            'author_betti0_error':abs(max(1,pb[0])-max(1,yb[0])),
            'author_betti1_error':abs(pb[1]-yb[1]),
            'author_cldice_defined':bool(ps.any() and ys.any() and precision+recall>0)}
def verify():
    import gudhi as gd
    rng=np.random.default_rng(41);masks=[np.zeros((7,8),bool),np.ones((7,8),bool),np.eye(7,dtype=bool)]
    ring=np.zeros((7,7),bool);ring[1:6,1:6]=True;ring[2:5,2:5]=False;masks.append(ring)
    masks += [rng.random((7,8))<p for p in (.1,.3,.5,.7,.9) for _ in range(20)]
    comparisons=[]
    for i,m in enumerate(masks):
        cc=gd.CubicalComplex(top_dimensional_cells=1-m.astype(int));cc.compute_persistence()
        # Correct level-set Betti numbers; reference's all-interval count examined separately.
        truth=tuple(cc.persistent_betti_numbers(0,0)[:2])
        upstream=tuple(cc.persistent_betti_numbers(np.inf,-np.inf)[:2])
        assert betti(m)==truth,(i,betti(m),truth)
        comparisons.append({'case':i,'ours':betti(m),'at_threshold0':truth,'reference_all_intervals':upstream})
    return {'cases':len(masks),'passed':True,'gudhi':gd.__version__,'reference_mismatch_cases':[r for r in comparisons if r['ours']!=r['reference_all_intervals']],
            'note':'Dice definition matches binary foreground Dice; topology comparison to published scores must disclose any reference mismatch'}
if __name__=='__main__':
    import json
    from pathlib import Path
    result=verify();root=Path(__file__).resolve().parents[1]
    (root/'auto_res_logs/results/test_metric_verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
