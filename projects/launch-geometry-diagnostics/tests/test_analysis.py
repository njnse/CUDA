import csv
import importlib.util
import io
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('analyze',Path(__file__).parents[1]/'scripts/analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def fixture():
    # Synthetic analysis fixtures only, not measured GPU data.
    rows=[]
    for n,r in sorted(a.SHAPES):
        for p in a.POLICIES:
            b=256 if p=='occupancy' else int(p[5:]);grid=min((n+b-1)//b,58*32)
            for i in range(9):rows.append([n,r,p,b,grid,58,1536,1536//b,i,20,5,2027,1+i/100,2+i/100])
    return rows


def stream(rows):
    f=io.StringIO();w=csv.writer(f);w.writerow(a.FIELDS);w.writerows(rows);f.seek(0);return f


class AnalysisTests(unittest.TestCase):
    def test_matched_and_dispersion(self):
        result=a.summarize(stream(fixture()))
        self.assertEqual(len(result),6)
        self.assertAlmostEqual(result[0]['policies']['occupancy']['timings']['device_ms']['median'],1.04)
        self.assertEqual(result[0]['policies']['occupancy']['predicted_occupancy_fraction'],1)
        self.assertAlmostEqual(result[0]['occupancy_over_best_fixed_device_ms'],1)

    def test_corrupt_evidence(self):
        for change in ('missing','duplicate','nan','negative','zero','seed','policy','grid','active','hardware','block'):
            with self.subTest(change=change):
                rows=fixture()
                if change=='missing':rows.pop()
                if change=='duplicate':rows.append(rows[0][:])
                if change=='nan':rows[0][12]='nan'
                if change=='negative':rows[0][13]=-1
                if change=='zero':rows[0][12]=0
                if change=='seed':rows[0][11]=0
                if change=='policy':rows[0][2]='other'
                if change=='grid':rows[0][4]+=1
                if change=='active':rows[0][7]=999
                if change=='hardware':rows[-1][6]=2048
                if change=='block':rows[0][3]=256
                with self.assertRaises(ValueError):a.summarize(stream(rows))

    def test_heuristic_can_be_slower(self):
        rows=fixture()
        for row in rows:
            if row[2]=='occupancy':row[12]=3
        self.assertTrue(all(r['occupancy_over_best_fixed_device_ms']>1 for r in a.summarize(stream(rows))))

    def test_schema(self):
        with self.assertRaises(ValueError):a.summarize(io.StringIO('x,y\n1,2\n'))


if __name__=='__main__':unittest.main()
