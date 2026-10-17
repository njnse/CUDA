import importlib.util
from pathlib import Path
import tempfile
import unittest
spec=importlib.util.spec_from_file_location("analysis",Path(__file__).parents[1]/"scripts/analyze.py")
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
class AnalysisTests(unittest.TestCase):
    def run_values(self,values):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"input.csv"
            p.write_text("n,mode,gpu_us,wall_us,submit_us\n"+"".join(f"257,graph,{x},{x},{x}\n" for x in values))
            return module.summarize(p)
    def test_median(self):
        self.assertEqual(self.run_values([1,2,99]),[(257,"graph",3,2,2,2,1,99)])
    def test_invalid(self):
        for value in ["nan","inf",0,-1]:
            with self.subTest(value=value),self.assertRaises(ValueError):self.run_values([value])
    def test_empty(self):
        with self.assertRaises(ValueError):self.run_values([])
if __name__=="__main__":unittest.main()
