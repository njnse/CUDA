#!/usr/bin/env python3
"""Require matched evidence and expose timing dispersion and numerical error."""
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
FIELDS = ['n','dataset','path','sample','repetitions','warmup','seed','device_ms','host_ms','abs_error','sum_abs','reference_sum']
SHAPES = {(n,d) for n in (257,65537,1048576) for d in ('positive','cancellation','dyadic')}
PATHS = ('tree_fp32','tree_fp64','cub_fp32')


def summarize(stream):
    reader=csv.DictReader(stream)
    if reader.fieldnames != FIELDS: raise ValueError('unexpected schema')
    groups=defaultdict(dict)
    references={}
    for row in reader:
        if None in row or any(v is None for v in row.values()): raise ValueError('ragged row')
        shape=int(row['n']),row['dataset']
        path=row['path']; sample=int(row['sample'])
        if shape not in SHAPES or path not in PATHS or not 0<=sample<9: raise ValueError('invalid case')
        if tuple(int(row[k]) for k in ('repetitions','warmup','seed')) != (30,5,2027): raise ValueError('unmatched settings')
        values={k:float(row[k]) for k in FIELDS[7:]}
        if any(not math.isfinite(v) for v in values.values()): raise ValueError('nonfinite value')
        if any(values[k]<=0 for k in ('device_ms','host_ms','sum_abs')) or values['abs_error']<0: raise ValueError('invalid metric')
        if abs(values['reference_sum'])>values['sum_abs']: raise ValueError('invalid oracle magnitude')
        reference=values['reference_sum'],values['sum_abs']
        if shape in references and reference != references[shape]: raise ValueError('unmatched oracle')
        references[shape]=reference
        if (path=='tree_fp64' or shape[1]=='positive') and values['abs_error']!=0: raise ValueError('exactness contract failed')
        gamma=shape[0]*2**-24/(1-shape[0]*2**-24)
        if values['abs_error']>gamma*values['sum_abs']: raise ValueError('error bound failed')
        key=(*shape,path)
        if sample in groups[key]: raise ValueError('duplicate sample')
        groups[key][sample]=values
    if set(groups)!={(n,d,p) for n,d in SHAPES for p in PATHS}: raise ValueError('incomplete sweep')
    result=[]
    for (n,dataset,path), samples in sorted(groups.items()):
        if set(samples)!=set(range(9)): raise ValueError('incomplete samples')
        timing={}
        for metric in ('device_ms','host_ms'):
            x=[r[metric] for r in samples.values()]
            q=statistics.quantiles(x,n=4,method='inclusive')
            timing[metric]={'median':statistics.median(x),'min':min(x),'max':max(x),'iqr':q[2]-q[0]}
        max_error=max(r['abs_error'] for r in samples.values())
        reference,absolute=references[n,dataset]
        result.append({'n':n,'dataset':dataset,'path':path,'samples':9,'timings':timing,
                       'max_abs_error':max_error,'max_error_over_sum_abs':max_error/absolute,
                       'reference_sum':reference,'sum_abs':absolute,
                       'cancellation_condition':absolute/abs(reference) if reference else None})
    return result


if __name__=='__main__':
    try:
        if len(sys.argv)!=2: raise ValueError('usage: analyze.py raw.csv')
        with open(sys.argv[1],newline='',encoding='utf-8') as f: print(json.dumps(summarize(f),indent=2,allow_nan=False))
    except (ValueError,OSError) as e:
        print('analysis failed: '+str(e),file=sys.stderr);sys.exit(1)
