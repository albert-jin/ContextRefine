"""Resumable public mirror download; preserve archive, source and per-file inventory."""
import argparse,hashlib,json,os,time,traceback,zipfile
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parents[1];LOG=ROOT/'auto_res_logs/icassp'
SOURCES={
 'fives':{'url':'https://www.kaggle.com/api/v1/datasets/download/nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation?datasetVersionNumber=6','expected_bytes':1763102639,'original':'https://doi.org/10.6084/m9.figshare.19688169.v1','mirror':'https://www.kaggle.com/datasets/nikitamanaenkov/fundus-image-dataset-for-vessel-segmentation','purpose':'CC BY 4.0 confirmed from original DOI DataCite metadata; cite Jin et al Scientific Data2022. Private academic vessel segmentation; preserve original600/200 split and disclose fixed validation split.'},
 'crack500':{'url':'https://www.kaggle.com/api/v1/datasets/download/pauldavid22/crack50020220509t090436z001?datasetVersionNumber=1','expected_bytes':1643195204,'original':'https://github.com/fyangneil/pavement-crack-detection','mirror':'https://www.kaggle.com/datasets/pauldavid22/crack50020220509t090436z001','purpose':'Private authorized academic segmentation experiments; no redistribution of dataset.'},
 'deepglobe':{'url':'https://www.kaggle.com/api/v1/datasets/download/balraj98/deepglobe-road-extraction-dataset?datasetVersionNumber=2','expected_bytes':4074676823,'original':'https://deepglobe.org/','mirror':'https://www.kaggle.com/datasets/balraj98/deepglobe-road-extraction-dataset','purpose':'Fetch and inspect dataset and its included license. Verify research-use terms before training; no redistribution.'}}
def atomic(p,d):q=p.with_suffix('.tmp');q.write_text(json.dumps(d,indent=2));q.replace(p)
def main():
    p=argparse.ArgumentParser();p.add_argument('dataset',choices=SOURCES);a=p.parse_args();src=SOURCES[a.dataset]
    out=ROOT/'data/icassp_downloads'/a.dataset;out.mkdir(parents=True,exist_ok=True);dest=out/'source.zip';partial=out/'source.zip.part'
    state={'dataset':a.dataset,'status':'starting','pid':os.getpid(),'created':time.time(),'source':src};status=LOG/(a.dataset+'_download.json');atomic(status,state)
    try:
        if not dest.exists():
            for attempt in range(1,5):
                offset=partial.stat().st_size if partial.exists() else 0
                headers={'Range':f'bytes={offset}-'} if offset else {}
                try:
                    with requests.get(src['url'],headers=headers,stream=True,timeout=(20,90)) as r:
                        r.raise_for_status();assert 'zip' in r.headers.get('Content-Type',''),r.headers
                        if offset and r.status_code!=206:offset=0
                        size=offset;last=time.time()
                        with partial.open('ab' if offset else 'wb') as f:
                            for block in r.iter_content(1024*1024):
                                if block:f.write(block);size+=len(block)
                                if time.time()-last>=5:
                                    state.update(status='downloading',bytes=size,expected_bytes=src['expected_bytes'],attempt=attempt,updated=time.time());atomic(status,state);last=time.time()
                    assert size==src['expected_bytes'],(size,src['expected_bytes']);partial.replace(dest);break
                except Exception as e:
                    state.update(last_error=repr(e),attempt=attempt);atomic(status,state)
                    if attempt==4:raise
                    time.sleep(5*attempt)
        state.update(status='verifying',bytes=dest.stat().st_size);atomic(status,state)
        h=hashlib.sha256()
        with dest.open('rb') as f:
            for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
        state['archive_sha256']=h.hexdigest();extract=(out/'raw').resolve();extract.mkdir(exist_ok=True)
        with zipfile.ZipFile(dest) as z:
            names=z.namelist();inventory=[{'path':i.filename,'bytes':i.file_size,'crc':i.CRC} for i in z.infolist()]
            assert all((extract/n).resolve().is_relative_to(extract) for n in names)
            atomic(out/'archive_inventory.json',inventory)
            # Crack500 archive also contains author predictions; extract only raw splits, masks and metadata.
            chosen=[n for n in names if not any(x in n.lower() for x in ('testresult','test_result','trainresult','model/','models/'))]
            state.update(status='extracting',files=len(chosen));atomic(status,state)
            for n in chosen:z.extract(n,extract)
        state.update(status='completed',finished=time.time(),extracted=str(extract),inventory=str(out/'archive_inventory.json'));atomic(status,state)
        atomic(out/'source_manifest.json',state);print(json.dumps(state))
    except BaseException as e:
        state.update(status='failed',error=repr(e));atomic(status,state);(LOG/(a.dataset+'_download_traceback.txt')).write_text(traceback.format_exc());raise
if __name__=='__main__':main()
