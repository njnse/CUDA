import csv
import math
import re
import statistics
import sys
from collections import defaultdict
def summarize(path):
    grouped=defaultdict(list)
    with open(path,newline="") as f:
        for row in csv.DictReader(f):
            values=[float(row[k]) for k in ("gpu_us","wall_us","submit_us")]
            if any(not math.isfinite(x) or x<=0 for x in values):
                raise ValueError("timings must be finite and positive")
            grouped[(int(row["n"]),row["mode"])].append(values)
    if not grouped:
        raise ValueError("empty benchmark")
    return [(n,mode,len(v),*[statistics.median(x[i] for x in v) for i in range(3)],
             min(x[1] for x in v),max(x[1] for x in v)) for (n,mode),v in sorted(grouped.items())]
if __name__=="__main__":
    if len(sys.argv)!=2: raise SystemExit("usage: analyze.py benchmark.csv")
    writer=csv.writer(sys.stdout)
    writer.writerow(["n","mode","samples","median_gpu_us","median_wall_us","median_submit_us","min_wall_us","max_wall_us"])
    writer.writerows(summarize(sys.argv[1]))
