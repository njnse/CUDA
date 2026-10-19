import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('worker',Path(__file__).parents[1]/'scripts/run_worker.py')
w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)


class RunnerConfigurationTests(unittest.TestCase):
    def test_invalid_output(self):
        for output in ('','relative/path',str(w.PROJECT),str(w.PROJECT/'src/probe')):
            with self.subTest(output=output),patch.dict(os.environ,{'RESULTS_DIR':output}):
                with self.assertRaises(ValueError):w.main()
        self.assertFalse((w.PROJECT/'src/probe').exists())

    def test_preserve_existing_artifact(self):
        # Only owned test fixtures are removed, beneath the project's build tree.
        build=w.ensure_build()
        with tempfile.TemporaryDirectory(dir=build) as d:
            target=Path(d)/'validation.json';target.write_text('preserve me')
            # An output directory under build is itself rejected before any writes.
            with patch.dict(os.environ,{'RESULTS_DIR':d}):
                with self.assertRaises(ValueError):w.main()
            self.assertEqual(target.read_text(),'preserve me')

    def test_manifest_scope(self):
        manifest=w.source_manifest()
        self.assertIn('src/pool.cu',manifest)
        self.assertTrue(all(not Path(p).is_absolute() for p in manifest))
        self.assertTrue(all(len(digest)==64 for digest in manifest.values()))
        self.assertTrue(all(p.split('/')[0] in ('src','include','scripts','tests') for p in manifest))


if __name__=='__main__':unittest.main()
