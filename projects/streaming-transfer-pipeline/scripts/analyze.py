#!/usr/bin/env python3
"""Validate a complete matched transfer sweep before summarizing timings."""
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
FIELDS=['n','chunk','path','sample','repetitions','warmup','seed','host_ms','logical_GBps']
SHAPES={(n,c) for n in (65537,1048579,4194307) for c in (4096,65536)}
PATHS=('pageable_serial','pinned_serial','pinned_pipeline')


def summarize(stream):
    reader=csv.DictReader(stream)
    if reader.fieldnames!=FIELDS: raise ValueError('unexpected schema')
    groups=defaultdict(dict)
    for row in reader:
        if None in row or any(v is None for v in row.values()): raise ValueError('ragged row')
        n,c,s=(int(row[k]) for k in ('n','chunk','sample'));path=row['path']
        if (n,c) not in SHAPES or path not in PATHS or not 0<=s<9: raise ValueError('unexpected case')
        if tuple(int(row[k]) for k in ('repetitions','warmup','seed'))!=(3,3,2027): raise ValueError('unmatched settings')
        elapsed,bandwidth=float(row['host_ms']),float(row['logical_GBps'])
        if any(not math.isfinite(v) or v<=0 for v in (elapsed,bandwidth)): raise ValueError('invalid timing or bandwidth')
        if not math.isclose(bandwidth,8*n/(elapsed*1e6),rel_tol=1e-8): raise ValueError('inconsistent logical bandwidth')
        key=n,c,path
        if s in groups[key]: raise ValueError('duplicate sample')
        groups[key][s]=elapsed
    if set(groups)!={(n,c,p) for n,c in SHAPES for p in PATHS}: raise ValueError('incomplete sweep')
    for group in groups.values():
        if set(group)!=set(range(9)): raise ValueError('incomplete samples')
    output=[]
    for n,c in sorted(SHAPES):
        timings={}
        for path in PATHS:
            x=list(groups[n,c,path].values());q=statistics.quantiles(x,n=4,method='inclusive')
            median=statistics.median(x)
            timings[path]={'median_ms':median,'min_ms':min(x),'max_ms':max(x),'iqr_ms':q[2]-q[0],
                           'logical_GBps_from_median':8*n/(median*1e6)}
        differences=[groups[n,c,'pinned_serial'][i]-groups[n,c,'pinned_pipeline'][i] for i in range(9)]
        output.append({'n':n,'chunk':c,'samples':9,'timings':timings,
                       'pinned_serial_over_pipeline':timings['pinned_serial']['median_ms']/timings['pinned_pipeline']['median_ms'],
                       'paired_pipeline_saving_ms':{'min':min(differences),'median':statistics.median(differences),'max':max(differences)},
                       'all_pairs_favor_pipeline':all(v>0 for v in differences)})
    return output


if __name__=='__main__':
    try:
        if len(sys.argv)!=2: raise ValueError('usage: analyze.py raw.csv')
        with open(sys.argv[1],newline='',encoding='utf-8') as f: print(json.dumps(summarize(f),indent=2,allow_nan=False))
    except (ValueError,OSError) as e:
        print('analysis failed: '+str(e),file=sys.stderr);sys.exit(1)
