"""Query physical GPUs immediately before a job; defer when headroom is insufficient.
This is a one-job launcher, not an unattended multi-host scheduler."""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

def snapshot():
    proc = subprocess.run(["nvidia-smi","--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu",
                           "--format=csv,noheader,nounits"],capture_output=True,text=True,check=True)
    rows=[]
    for r in csv.reader(proc.stdout.splitlines()):
        idx,uuid,name,total,used,util = [s.strip() for s in r]
        rows.append(dict(index=int(idx),uuid=uuid,name=name,total_mib=int(total),
                         used_mib=int(used),utilization=int(util)))
    return rows

def eligible(rows,reserve):
    return sorted([r for r in rows if r["index"] in (0,1,2,3)
                   and r["used_mib"]/r["total_mib"] < .70
                   and (r["used_mib"]+reserve)/r["total_mib"] < .80],
                  key=lambda r:r["used_mib"]/r["total_mib"])

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reserve-mib",type=int,required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--launch",action="store_true")
    p.add_argument("command",nargs=argparse.REMAINDER)
    args=p.parse_args()
    if args.reserve_mib<=0: p.error("reserve-mib must be positive")
    rows=snapshot()
    candidates=eligible(rows,args.reserve_mib)
    record={"timestamp":datetime.now(timezone.utc).isoformat(),"gpus":rows,
            "reserve_mib":args.reserve_mib,"eligible_indices":[r["index"] for r in candidates],
            "status":"inspection_only"}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists(): raise FileExistsError(args.output)
    if args.launch:
        if not candidates: record["status"]="deferred_no_headroom"
        else: record["status"]="launch_requested"
    args.output.write_text(json.dumps(record,indent=2),encoding="utf-8")
    print(json.dumps(record,indent=2),flush=True)
    if not args.launch:return
    if not candidates:sys.exit(75)
    cmd=args.command
    if cmd and cmd[0]=="--":cmd=cmd[1:]
    if not cmd:p.error("launch requires a command")
    # Use UUID so the selected physical GPU is unambiguous.
    env=os.environ.copy()
    root=Path(__file__).resolve().parents[1]
    for key,relative in {'TMPDIR':'tmp','XDG_CACHE_HOME':'cache','TORCH_HOME':'cache/torch','CUDA_CACHE_PATH':'cache/cuda'}.items():
        directory=root/relative;directory.mkdir(parents=True,exist_ok=True);env[key]=str(directory)
    env["CUDA_VISIBLE_DEVICES"]=candidates[0]["uuid"]
    env["SCNP_GPU_GUARDED"]="1"
    env["SCNP_GPU_RESERVE_MIB"]=str(args.reserve_mib)
    # CUDA runtime must be checked using the actual Python command interpreter.
    if Path(cmd[0]).stem.lower() not in ("python","python3"):
        raise ValueError("Only explicit Python training commands supported")
    check=subprocess.run([cmd[0],"-c","import torch; assert torch.cuda.is_available(), 'CUDA PyTorch unavailable'"],
                         env=env,capture_output=True,text=True)
    record["cuda_check"]={"returncode":check.returncode,"stderr":check.stderr}
    if check.returncode:
        record["status"]="blocked_cuda_runtime"
        args.output.write_text(json.dumps(record,indent=2),encoding="utf-8")
        print(check.stderr);sys.exit(check.returncode)
    # Refresh snapshot after runtime check to avoid stale preflight.
    fresh=eligible(snapshot(),args.reserve_mib)
    if candidates[0]["uuid"] not in [r["uuid"] for r in fresh]:
        record["status"]="deferred_usage_changed"
        args.output.write_text(json.dumps(record,indent=2),encoding="utf-8");sys.exit(75)
    record["command"]=cmd
    record["selected_gpu"]=candidates[0]
    record["status"]="running"
    args.output.write_text(json.dumps(record,indent=2),encoding="utf-8")
    with args.output.with_suffix(".stdout.log").open("w",encoding="utf-8") as out:
        job=subprocess.run(cmd,env=env,stdout=out,stderr=subprocess.STDOUT)
    record["returncode"]=job.returncode
    record["status"]="completed" if job.returncode==0 else "failed"
    record["finished_at"]=datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(record,indent=2),encoding="utf-8")
    sys.exit(job.returncode)

if __name__=="__main__":main()
