"""Pinned TopoMortar train/val downloader with Git blob and SHA256 verification."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
def fetch(url):
    for attempt in range(4):
        try:
            req=urllib.request.Request(url,headers={"User-Agent":"SCNP-research-reproduction"})
            with urllib.request.urlopen(req,timeout=45) as r:return r.read()
        except Exception:
            if attempt==3:raise
            time.sleep(2**attempt)
def main():
    tree=json.loads(fetch("https://api.github.com/repos/jmlipman/TopoMortar/git/trees/main?recursive=1"))
    if tree.get("truncated"):raise RuntimeError("Truncated tree")
    sha=tree["sha"]
    dest=ROOT/"data/TopoMortar"
    dest.mkdir(parents=True,exist_ok=True)
    (dest/"upstream_tree.json").write_text(json.dumps(tree,indent=2))
    metadata={"README.md","LICENSE","dataset/splits.yaml","dataset/splits_small.yaml","dataset/dataset_types.txt",
              "config/template_topomortar.yaml","lib/data/TopoMortar.py"}
    entries=[e for e in tree["tree"] if e["type"]=="blob" and (
        e["path"] in metadata or any(e["path"].startswith(f"dataset/{s}/{k}/")
        for s in ("train","val") for k in ("images","accurate")))]
    def download(e):
        path=dest/e["path"]
        expected=e["sha"]
        def gitsha(data):return hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
        data=path.read_bytes() if path.exists() else b""
        url=f"https://raw.githubusercontent.com/jmlipman/TopoMortar/{sha}/{e['path']}"
        if gitsha(data)!=expected:data=fetch(url)
        if gitsha(data)!=expected:raise ValueError(f"Hash mismatch {e['path']}")
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
        return {"path":e["path"],"url":url,"git_blob":expected,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)}
    records=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for r in pool.map(download,entries):
            records.append(r)
            print(f"verified {len(records)}/{len(entries)} {r['path']}",flush=True)
    record={"repository":"https://github.com/jmlipman/TopoMortar","commit":sha,
            "scope":"train/val accurate labels only; test intentionally not downloaded",
            "files":records}
    (dest/"download_manifest.json").write_text(json.dumps(record,indent=2))
    print(json.dumps({"complete":True,"files":len(records),"bytes":sum(x["bytes"] for x in records),"commit":sha}))
if __name__=="__main__":main()

