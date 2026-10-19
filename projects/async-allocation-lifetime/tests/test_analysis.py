import csv
import importlib.util
import io
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('analyze',Path(__file__).parents[1]/'scripts/analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def fixture():
    # Synthetic parser fixtures only, never published as measurements.
    return [[n,p,i,20,3,2027,(1 if p=='async_retain' else 2)+i/100,
             2097152 if p=='async_retain' else 0,0]
            for n in a.SIZES for p in a.PATHS for i in range(9)]


def stream(rows):
    f=io.StringIO();w=csv.writer(f);w.writerow(a.FIELDS);w.writerows(rows);f.seek(0);return f


class AnalysisTests(unittest.TestCase):
    def test_complete_and_memory_cost(self):
        result=a.summarize(stream(fixture()))
        self.assertEqual(len(result),3)
        self.assertAlmostEqual(result[0]['timings']['async_retain']['median_ms'],1.04)
        self.assertEqual(result[0]['allocated_bytes'],1280)
        self.assertEqual(result[0]['timings']['async_retain']['reserved_bytes_min'],2097152)

    def test_corrupt_evidence(self):
        for change in ('missing','duplicate','nan','negative','zero','seed','path','size','reserved','used','legacy'):
            with self.subTest(change=change):
                rows=fixture()
                if change=='missing':rows.pop()
                if change=='duplicate':rows.append(rows[0][:])
                if change=='nan':rows[0][6]='nan'
                if change=='negative':rows[0][6]=-1
                if change=='zero':rows[0][6]=0
                if change=='seed':rows[0][5]=0
                if change=='path':rows[0][1]='other'
                if change=='size':rows[0][0]=1
                if change=='reserved':rows[0][7]=-1
                if change=='used':rows[0][8]=4
                if change=='legacy':rows[0][7]=4096
                with self.assertRaises(ValueError):a.summarize(stream(rows))

    def test_adverse_result(self):
        rows=fixture()
        for row in rows:
            if row[1]=='async_retain':row[6]=3
        self.assertTrue(all(r['release_over_retained']<1 for r in a.summarize(stream(rows))))

    def test_schema(self):
        with self.assertRaises(ValueError):a.summarize(io.StringIO('a,b\n1,2\n'))


if __name__=='__main__':unittest.main()
