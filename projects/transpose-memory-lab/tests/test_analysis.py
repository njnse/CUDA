import importlib.util
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("analysis", Path(__file__).parents[1] / "scripts/analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


class AnalysisTests(unittest.TestCase):
    def run_csv(self, values):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            path.write_text("variant,rows,cols,ms_per_launch\n" + "".join(
                f"padded,31,33,{v}\n" for v in values))
            return analysis.summarize([path])

    def test_median_not_mean(self):
        self.assertEqual(self.run_csv([1, 2, 99]), [("padded", 31, 33, 3, 2, 1, 99)])

    def test_reject_invalid_measurements(self):
        for value in ["nan", "inf", "0", "-1"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.run_csv([value])

    def test_reject_empty(self):
        with self.assertRaises(ValueError):
            self.run_csv([])


if __name__ == "__main__":
    unittest.main()
