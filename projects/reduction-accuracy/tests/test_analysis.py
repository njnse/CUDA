import csv
import importlib.util
import io
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('analyze',Path(__file__).parents[1]/'scripts/analyze.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def fixture():
    # Synthetic parser fixtures, never presented as measured GPU evidence.
    return [[n,d,p,i,30,5,2027,1+i/100,2+i/100,0,100,2]
            for n,d in sorted(a.SHAPES) for p in a.PATHS for i in range(9)]


def stream(rows):
    f=io.StringIO();w=csv.writer(f);w.writerow(a.FIELDS);w.writerows(rows);f.seek(0);return f


class AnalysisTests(unittest.TestCase):
    def test_valid_and_dispersion(self):
        result=a.summarize(stream(fixture()))
        self.assertEqual(len(result),27)
        self.assertAlmostEqual(result[0]['timings']['device_ms']['median'],1.04)
        self.assertGreater(result[0]['timings']['device_ms']['iqr'],0)

    def test_corrupt_evidence(self):
        for change in ('missing','duplicate','nan','negative','seed','reference','path','zero','fp64','positive'):
            with self.subTest(change=change):
                rows=fixture()
                if change=='missing':rows.pop()
                if change=='duplicate':rows.append(rows[0][:])
                if change=='nan':rows[0][7]='nan'
                if change=='negative':rows[0][9]=-1
                if change=='seed':rows[0][6]=0
                if change=='reference':rows[0][11]=3
                if change=='path':rows[0][2]='other'
                if change=='zero':rows[0][7]=0
                if change in ('fp64','positive'):
                    for row in rows:
                        if (row[2]=='tree_fp64' if change=='fp64' else row[1]=='positive'):
                            row[9]=0.01;break
                with self.assertRaises(ValueError):a.summarize(stream(rows))

    def test_schema(self):
        with self.assertRaises(ValueError):a.summarize(io.StringIO('a,b\n1,2\n'))

    def test_zero_reference(self):
        rows=fixture()
        for row in rows:row[11]=0
        self.assertTrue(all(r['cancellation_condition'] is None for r in a.summarize(stream(rows))))

    def test_fp32_cancellation_error_is_reported(self):
        rows=fixture()
        for row in rows:
            if row[1]=='cancellation' and row[2]=='tree_fp32':row[9]=0.00001
        result=a.summarize(stream(rows))
        self.assertTrue(any(r['max_abs_error']>0 for r in result))


if __name__=='__main__':unittest.main()
