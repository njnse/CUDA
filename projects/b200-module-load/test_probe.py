import subprocess
import sys
import unittest

from probe import GUARD, input_values, oracle, validate_output


class OracleTest(unittest.TestCase):
    def test_wrap_and_index(self):
        self.assertEqual(oracle([0, 1, 0xFFFFFFFF]), [0, 4, 0xFFFFFFFF])
        self.assertEqual(oracle([0] * 18)[15:], [15, 16, 0])

    def test_tail_guard_and_value_error(self):
        validate_output([7], [7, GUARD])
        for malformed in ([7], [7, 0], [8, GUARD], [7, GUARD, GUARD]):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                validate_output([7], malformed)

    def test_deterministic_inputs(self):
        self.assertEqual(input_values(3), [17, 2654435778, 1013904243])

    def test_cli_rejects_invalid_length_before_cuda(self):
        for n in ('0', '65538'):
            with self.subTest(n=n):
                result = subprocess.run([sys.executable, 'probe.py', 'unused.ptx', n],
                                        capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('n must be in', result.stderr)
