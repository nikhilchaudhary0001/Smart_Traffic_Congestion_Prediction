"""Prepare historical road observations for TRAFFICIQ without touching the trained model.
Run once: .\.venv\Scripts\python.exe prepare_trafficiq_data.py
"""
from pathlib import Path
import zipfile, json, csv, gzip, re
ROOT=Path(__file__).resolve().parent
ZIP=ROOT/'archive (1).zip'
OUT=ROOT/'trafficiq_data'
OUT.mkdir(exist_ok=True)
if not ZIP.exists(): raise SystemExit('Missing archive (1).zip in project folder')
with zipfile.ZipFile(ZIP) as z, gzip.open(OUT/'observations.csv.gz','wt',newline='',encoding='utf-8') as out:
    writer=csv.writer(out)
    writer.writerow(['segment_id','date','counts'])
    metadata={}
    names=sorted(n for n in z.namelist() if '/probe_counts/geojson/' in n and n.endswith('.geojson'))
    for day,n in enumerate(names):
        m=re.search(r'2024-\d\d-\d\d',n)
        if not m: continue
        date=m.group(0)
        data=json.load(z.open(n))
        meta=data['features'][0]['properties']
        times={int(t['@id']):int(t['name'].split(':')[0]) for t in meta['timeSets']}
        for feature in data['features'][1:]:
            p=feature.get('properties',{})
            sid=str(p.get('newSegmentId') or p.get('segmentId'))
            if not sid: continue
            counts=[None]*24
            for rec in p.get('segmentProbeCounts',[]):
                hour=times.get(rec.get('timeSet'))
                if hour is not None and 0<=hour<24: counts[hour]=rec.get('probeCount')
            writer.writerow([sid,date,json.dumps(counts,separators=(',',':'))])
            if day==0:
                geom=feature.get('geometry') or {}
                metadata[sid]={'name':p.get('streetName') or 'Unnamed road','speedLimit':p.get('speedLimit'), 'frc':p.get('frc'),'distance':p.get('distance'),'coordinates':geom.get('coordinates',[]) if geom.get('type')=='LineString' else []}
        print(f'Processed {date} ({day+1}/{len(names)})',flush=True)
    (OUT/'roads.json').write_text(json.dumps(metadata,separators=(',',':')),encoding='utf-8')
print('Prepared TRAFFICIQ data. Your original ZIP and model were not modified.')
