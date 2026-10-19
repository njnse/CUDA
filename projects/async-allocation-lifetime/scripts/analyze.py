#!/usr/bin/env python3
"""Summarize only complete, matched allocation-lifetime samples."""
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
FIELDS=['n','path','sample','repetitions','warmup','seed','host_ms','reserved_bytes','used_bytes']
SIZES=(256,65537,1048576)
PATHS=('legacy','async_release','async_retain')


def summarize(stream):
    reader=csv.DictReader(stream)
    if reader.fieldnames!=FIELDS:raise ValueError('unexpected CSV schema')
    groups=defaultdict(dict)
    for row in reader:
        if None in row or any(v is None for v in row.values()):raise ValueError('ragged row')
        n,s=int(row['n']),int(row['sample']);p=row['path']
        if n not in SIZES or p not in PATHS or not 0<=s<9:raise ValueError('unexpected case')
        if tuple(int(row[k]) for k in ('repetitions','warmup','seed'))!=(20,3,2027):raise ValueError('mismatched settings')
        t=float(row['host_ms']);reserved=int(row['reserved_bytes']);used=int(row['used_bytes'])
        if not math.isfinite(t) or t<=0:raise ValueError('invalid timing')
        if not 0<=reserved<2**64 or used!=0:raise ValueError('invalid or live pool bytes')
        if p=='legacy' and reserved!=0:raise ValueError('legacy pool metric must be zero sentinel')
        if s in groups[n,p]:raise ValueError('duplicate sample')
        groups[n,p][s]=(t,reserved)
    if set(groups)!={(n,p) for n in SIZES for p in PATHS}:raise ValueError('incomplete sweep')
    for group in groups.values():
        if set(group)!=set(range(9)):raise ValueError('incomplete samples')
    results=[]
    for n in SIZES:
        timing={}
        for p in PATHS:
            x=[v[0] for v in groups[n,p].values()];r=[v[1] for v in groups[n,p].values()]
            q=statistics.quantiles(x,n=4,method='inclusive')
            timing[p]={'median_ms':statistics.median(x),'min_ms':min(x),'max_ms':max(x),'iqr_ms':q[2]-q[0],
                       'reserved_bytes_min':min(r),'reserved_bytes_max':max(r)}
        paired=[groups[n,'async_release'][i][0]-groups[n,'async_retain'][i][0] for i in range(9)]
        results.append({'n':n,'allocated_bytes':(n+64)*4,'samples':9,'timings':timing,
                        'legacy_over_retained':timing['legacy']['median_ms']/timing['async_retain']['median_ms'],
                        'release_over_retained':timing['async_release']['median_ms']/timing['async_retain']['median_ms'],
                        'paired_retention_saving_ms':{'min':min(paired),'median':statistics.median(paired),'max':max(paired)}})
    return results


if __name__=='__main__':
    try:
        if len(sys.argv)!=2:raise ValueError('usage: analyze.py raw.csv')
        with open(sys.argv[1],newline='',encoding='utf-8') as f:print(json.dumps(summarize(f),indent=2,allow_nan=False))
    except (OSError,ValueError) as e:
        print('analysis failed: '+str(e),file=sys.stderr);sys.exit(1)
