import hashlib,json,os,shlex,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];REPO=Path(__file__).resolve().parents[1]
OUT=ROOT/'auto_res_logs/crack500_context_v8_20260918'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert not (OUT/'round_results.json').exists()
    rows=[]
    warm=REPO/'auto_res_logs/runs/crack_context_base_s0/best.pt'
    local=REPO/'auto_res_logs/runs/crack_context_local_refine_s0'
    dino=REPO/'auto_res_logs/runs/crack_context_dino_context_s0'
    a,b=read(local/'config.json'),read(dino/'config.json')
    keys=['dataset','manifest_sha256','seed','steps','batch','workers','val_every','warm','crop','selection','test','supervision','optimizer','scheduler','clip_norm','precision','paired_js_coefficient']
    checks={k:a[k]==b[k] for k in keys}
    ia,ib=read(local/'initialization.json'),read(dino/'initialization.json')
    checks['initial_head_equal']=ia['trainable_sha256']==ib['trainable_sha256']
    checks['same_frozen_local_checkpoint']=ia['base_checkpoint_sha256']==ib['base_checkpoint_sha256']==sha(warm)
    checks['both_frozen_unchanged']=all(read(p/'frozen_verification.json')['passed'] for p in [local,dino])
    al=[json.loads(x) for x in (local/'train.jsonl').read_text().splitlines()]
    bl=[json.loads(x) for x in (dino/'train.jsonl').read_text().splitlines()]
    checks['same_recorded_steps']=[x['step'] for x in al]==[x['step'] for x in bl]
    checks['same_sample_stream']=len(al)==len(bl) and all(x['sample_ids']==y['sample_ids'] for x,y in zip(al,bl))
    checks['same_augmentation_stream']=len(al)==len(bl) and all(x['diag']['coefficient']==y['diag']['coefficient'] and x['diag']['occlusion_fraction']==y['diag']['occlusion_fraction'] for x,y in zip(al,bl))
    (OUT/'matched_comparison_audit.json').write_text(json.dumps({'time':time.time(),'checks':checks,'passed':all(checks.values()),'logged_batches':len(al)},indent=2))
    assert all(checks.values()),checks
    for stage in ['base','local_refine','dino_context']:
        ident='crack_context_'+stage+'_s0';run=REPO/'auto_res_logs/runs'/ident
        assert read(run/'status.json')['status']=='completed'
        cmd=[str(ROOT/'env/bin/python'),str(REPO/'research/train_crack500_context.py'),'--stage',stage,'--run-id',ident,'--steps','12000' if stage=='base' else '3000','--val-every','1200' if stage=='base' else '300','--mode','test']
        if stage!='base':cmd+=['--warm',str(warm)]
        with (OUT/'finalize_commands.jsonl').open('a') as f:f.write(json.dumps({'time':time.time(),'command':cmd,'cwd':str(REPO)})+'\n')
        with (OUT/(ident+'_test.stdout.log')).open('x') as f:
            subprocess.run(cmd,cwd=REPO,env=os.environ.copy(),stdout=f,stderr=subprocess.STDOUT,check=True)
        p=run/'test/summary.json';d=read(p);assert len(d['per_image'])==200
        assert read(run/'validation_replay.json')['passed']
        rows.append({'run_id':ident,'method':stage,'mean':d['mean'],'test_records':200,'selected_step':d['selected_step'],'checkpoint':str(run/'best.pt'),'checkpoint_sha256':d['checkpoint_sha256'],'source':str(p.relative_to(ROOT)),'source_sha256':sha(p),'validation_replay_passed':True})
    old=read(ROOT/'icassp_main_results/main_results.json')['rows']
    baseline=next(r for r in old if r['dataset'].lower()=='crack500' and r['method']=='SCNP')
    budget=next(r for r in old if r['dataset'].lower()=='crack500' and r['method']=='Structure gradient budget')
    for r in rows:
        r['delta_historical_SCNP_pp']=100*(r['mean']['dice']-baseline['mean']['dice'])
        r['delta_historical_budget_pp']=100*(r['mean']['dice']-budget['mean']['dice'])
        r['delta_new_convnext_base_pp']=100*(r['mean']['dice']-rows[0]['mean']['dice'])
        r['delta_matched_local_refine_pp']=100*(r['mean']['dice']-rows[1]['mean']['dice'])
    result={'created':time.time(),'dataset':'Crack500','status':'completed','rows':rows,'historical_scnp':baseline,'historical_budget':budget,
        'protocol':'same fixed cleaned243/49/200;native-label200test;threshold0.5;noTTA','seed':0,
        'interpretation':'Historical ResNet baseline comparison is a full-recipe comparison. DINO-versus-local comparison shares frozen local backbone,initial head,data streams,training budget and losses; DINO adds pretrained context and projection parameters. No new-method or SOTA claim follows from scores alone.'}
    (OUT/'round_results.json').write_text(json.dumps(result,indent=2))
    lines=['# Crack500 v8 transfer results','', 'All runs use seed 0 and the full 200-image test set. Model selection is fixed and replayed on 49 validation images before testing.','',
        '| Method | Dice% | IoU% | clDice% | Betti-0 error | Betti-1 error | vs. historical SCNP (pp) | vs. ConvNeXt base (pp) | vs. matched local refinement (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        m=r['mean'];lines.append(f"| {r['method']} | {100*m['dice']:.4f} | {100*m['foreground_iou']:.4f} | {100*m['cldice']:.4f} | {m['betti0_error']:.4f} | {m['betti1_error']:.4f} | {r['delta_historical_SCNP_pp']:+.4f} | {r['delta_new_convnext_base_pp']:+.4f} | {r['delta_matched_local_refine_pp']:+.4f} |")
    lines+=['','Historical SCNP achieved 71.0452303873%; the earlier structure gradient budget method achieved 71.6089163996%. All test results, including negative results, are retained.','',
        'Historical SCNP uses ResNet34, whereas the new configuration uses ConvNeXt and additional DINO pretraining; historical differences therefore compare complete training recipes. The matched local_refine and dino_context runs share base weights, initialization, data, augmentations, training budget, and model-selection rules. DINO adds pretrained resources, so the difference cannot be attributed entirely to a new structural contribution.','',
        'This uses the same v8 context refinement head and paired SCNP/JS/occlusion training as TopoMortar. Only data loading, the dataset-specific local backbone, and Crack500 native-resolution validation selection differ. No TopoMortar segmentation weights are used.']
    (OUT/'RESULTS.md').write_text('\n'.join(lines)+'\n')
    (OUT/'finalize_status.json').write_text(json.dumps({'status':'completed','time':time.time(),'models':len(rows)}))
    print(json.dumps({'completed':True,'rows':rows}),flush=True)
if __name__=='__main__':main()
