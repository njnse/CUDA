#!/usr/bin/env python3
"""Reject incomplete/mismatched launch experiments before comparing policies."""
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
FIELDS=['n','rounds','policy','block','grid','sm_count','max_threads_per_sm','active_blocks_per_sm',
        'sample','repetitions','warmup','seed','device_ms','host_ms']
SHAPES={(n,r) for n in (257,65537,1048576) for r in (1,64)}
POLICIES=('fixed128','fixed256','fixed512','occupancy')


def summarize(stream):
    reader=csv.DictReader(stream)
    if reader.fieldnames!=FIELDS:raise ValueError('unexpected schema')
    groups=defaultdict(dict);configurations={};hardware=None;recommendation=None
    for row in reader:
        if None in row or any(v is None for v in row.values()):raise ValueError('ragged row')
        n,rounds,sample=(int(row[k]) for k in ('n','rounds','sample'));policy=row['policy']
        if (n,rounds) not in SHAPES or policy not in POLICIES or not 0<=sample<9:raise ValueError('invalid case')
        if tuple(int(row[k]) for k in ('repetitions','warmup','seed'))!=(20,5,2027):raise ValueError('unmatched settings')
        block,grid,sms,threads,active=(int(row[k]) for k in ('block','grid','sm_count','max_threads_per_sm','active_blocks_per_sm'))
        if not 32<=block<=1024 or block%32 or sms<=0 or threads<=0 or threads%32 or active<=0 or active*block>threads:
            raise ValueError('invalid occupancy metadata')
        if policy!='occupancy' and block!=int(policy[5:]):raise ValueError('fixed policy changed block')
        if grid!=min((n+block-1)//block,sms*32):raise ValueError('incorrect grid')
        if hardware is not None and hardware!=(sms,threads):raise ValueError('mixed devices')
        hardware=sms,threads
        if policy=='occupancy':
            if recommendation is not None and recommendation!=block:raise ValueError('changed recommendation')
            recommendation=block
        elapsed=tuple(float(row[k]) for k in ('device_ms','host_ms'))
        if any(not math.isfinite(v) or v<=0 for v in elapsed):raise ValueError('invalid timing')
        key=n,rounds,policy;configuration=block,grid,sms,threads,active
        if key in configurations and configurations[key]!=configuration:raise ValueError('changed launch configuration')
        configurations[key]=configuration
        if sample in groups[key]:raise ValueError('duplicate sample')
        groups[key][sample]=elapsed
    if set(groups)!={(n,r,p) for n,r in SHAPES for p in POLICIES}:raise ValueError('incomplete sweep')
    for group in groups.values():
        if set(group)!=set(range(9)):raise ValueError('incomplete samples')
    result=[]
    for n,rounds in sorted(SHAPES):
        policies={}
        for p in POLICIES:
            b,g,sms,threads,active=configurations[n,rounds,p]
            timing={}
            for index,label in enumerate(('device_ms','host_ms')):
                x=[v[index] for v in groups[n,rounds,p].values()];q=statistics.quantiles(x,n=4,method='inclusive')
                timing[label]={'median':statistics.median(x),'min':min(x),'max':max(x),'iqr':q[2]-q[0]}
            policies[p]={'block':b,'grid':g,'active_blocks_per_sm':active,
                         'predicted_occupancy_fraction':active*b/threads,'timings':timing}
        best=min(POLICIES[:3],key=lambda p:policies[p]['timings']['device_ms']['median'])
        heuristic=policies['occupancy']['timings']['device_ms']['median']
        paired=[groups[n,rounds,'fixed256'][i][0]-groups[n,rounds,'occupancy'][i][0] for i in range(9)]
        result.append({'n':n,'rounds':rounds,'samples':9,'policies':policies,'best_observed_fixed_device_policy':best,
                       'occupancy_over_best_fixed_device_ms':heuristic/policies[best]['timings']['device_ms']['median'],
                       'fixed256_over_occupancy_device_ms':policies['fixed256']['timings']['device_ms']['median']/heuristic,
                       'paired_occupancy_saving_device_ms':{'min':min(paired),'median':statistics.median(paired),'max':max(paired)}})
    return result


if __name__=='__main__':
    try:
        if len(sys.argv)!=2:raise ValueError('usage: analyze.py raw.csv')
        with open(sys.argv[1],newline='',encoding='utf-8') as f:print(json.dumps(summarize(f),indent=2,allow_nan=False))
    except (ValueError,OSError) as e:
        print('analysis failed: '+str(e),file=sys.stderr);sys.exit(1)
