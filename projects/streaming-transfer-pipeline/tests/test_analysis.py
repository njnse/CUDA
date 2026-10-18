import csv
import importlib.util
import io
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('analyze',Path(__file__).parents[1]/'scripts/analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def fixture():
    # Synthetic parser fixtures; not GPU measurements.
    rows=[]
    for n,c in sorted(a.SHAPES):
        for p in a.PATHS:
            for i in range(9):
                ms=(1 if p=='pinned_pipeline' else 2)+i/100
                rows.append([n,c,p,i,3,3,2027,ms,8*n/(ms*1e6)])
    return rows


def stream(rows):
    f=io.StringIO();w=csv.writer(f);w.writerow(a.FIELDS);w.writerows(rows);f.seek(0);return f


class AnalysisTests(unittest.TestCase):
    def test_matched(self):
        result=a.summarize(stream(fixture()))
        self.assertEqual(len(result),6)
        self.assertTrue(all(r['all_pairs_favor_pipeline'] for r in result))
        self.assertAlmostEqual(result[0]['timings']['pinned_pipeline']['median_ms'],1.04)
        self.assertGreater(result[0]['timings']['pinned_pipeline']['iqr_ms'],0)

    def test_bad_evidence(self):
        for change in ('missing','duplicate','nan','negative','zero','seed','shape','path','bandwidth'):
            with self.subTest(change=change):
                rows=fixture()
                if change=='missing':rows.pop()
                if change=='duplicate':rows.append(rows[0][:])
                if change=='nan':rows[0][7]='nan'
                if change=='negative':rows[0][7]=-1
                if change=='zero':rows[0][7]=0
                if change=='seed':rows[0][6]=0
                if change=='shape':rows[0][0]=1
                if change=='path':rows[0][2]='other'
                if change=='bandwidth':rows[0][8]=1e9
                with self.assertRaises(ValueError):a.summarize(stream(rows))

    def test_no_pipeline_gain(self):
        rows=fixture()
        for row in rows:
            if row[2]=='pinned_pipeline':row[7]=3;row[8]=8*row[0]/(3e6)
        result=a.summarize(stream(rows))
        self.assertTrue(all(r['pinned_serial_over_pipeline']<1 for r in result))
        self.assertTrue(all(not r['all_pairs_favor_pipeline'] for r in result))

    def test_schema(self):
        with self.assertRaises(ValueError):a.summarize(io.StringIO('x,y\n1,2\n'))


if __name__=='__main__':unittest.main()
