import unittest

from prepare_router_release_probe import transform


class RouterReleasePatchTests(unittest.TestCase):
    def test_only_inserts_fence_before_the_single_release(self):
        source = 'before\n                load_pipeline.consumer_release(consumer_state)\nafter\n'
        self.assertEqual(transform(source),
                         'before\n                cute.arch.fence_acq_rel_cta()\n'
                         '                load_pipeline.consumer_release(consumer_state)\nafter\n')

    def test_missing_or_ambiguous_site_fails(self):
        for source in ('', '                load_pipeline.consumer_release(consumer_state)\n' * 2):
            with self.assertRaises(RuntimeError):
                transform(source)


if __name__ == '__main__':
    unittest.main()
