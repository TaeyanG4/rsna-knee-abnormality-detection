"""Standard-library checks for the EXP183 frozen schedule and runner derivation."""

import unittest

from tools import exp183_multiscale_5fold as exp


class ContractTests(unittest.TestCase):
    def test_serial_four_arm_schedule_and_budget(self):
        self.assertEqual([item['tag'] for item in exp.queue()],
                         [exp.tag(fold) for fold in range(1, 5)])
        self.assertTrue(all(item['continue_on_failure'] is False
                            for item in exp.queue()))
        self.assertLessEqual(exp.ARM_CAP * len(exp.FOLDS), exp.TOTAL_CAP)
        self.assertEqual(exp.TOTAL_CAP, 4.5 * 3600)
        with self.assertRaisesRegex(ValueError, 'fold must be 1-4'):
            exp.tag(0)

    def test_derived_runner_and_remote_plan(self):
        source = exp.runner_source()
        self.assertIn('from tools import exp183_multiscale_5fold as guard', source)
        self.assertNotIn('from tools import exp179_multiscale as guard', source)
        self.assertIn('guard.stamp_outputs(out, protocol)', source)
        plan = exp.vast_plan()
        self.assertIn('PYTHONPATH=/workspace/exp179', plan)
        self.assertIn('max_gpu_seconds=16200', plan)
        self.assertIn('tools/gpu_queue.py queue.json --foreground', plan)
        with self.assertRaisesRegex(ValueError, 'absolute /workspace'):
            exp.vast_plan('../outside')


if __name__ == '__main__':
    unittest.main()
