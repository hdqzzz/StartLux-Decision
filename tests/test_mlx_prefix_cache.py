"""Behavior checks for request-scoped, exact-token prefix reuse on a hybrid model."""
import asyncio
import unittest
from tests.test_mlx_backend import HAVE_MLX, np
from tests.test_mlx_backend import tiny_folder, FakeTokenizer, run
from startlux_decide.backends.mlx import MLXBackend

@unittest.skipUnless(HAVE_MLX, "mlx-vlm is not installed")
class PrefixCacheTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.be = MLXBackend(tiny_folder(False), FakeTokenizer(), False, prefill_chunk=1024)
        run(cls.be.start())
        cls.be.prefix_cache = True
        cls.prompts = ['Evidence: ' + 'a' * 300 + '\nQuestion: ' + q for q in ['route?', 'topic?']]
        cls.letters = [40, 41, 42, 43]

    @classmethod
    def tearDownClass(cls):
        run(cls.be.close())

    def test_split_matches_cold_and_repeated_branches_do_not_mutate_snapshot(self):
        lane = run(self.be.share_prefix(self.prompts))
        self.assertIsNotNone(lane)
        cold = [run(self.be.readout(p, self.letters)) for p in self.prompts]
        first = [run(lane.readout(p, self.letters)) for p in self.prompts]
        for a, b in zip(cold, first):
            np.testing.assert_allclose(a, b, atol=2e-3, rtol=2e-4)
        # Run a much longer branch between repeats, crossing KV allocation boundaries.
        run(lane.readout(self.prompts[0] + 'b' * 400, self.letters))
        np.testing.assert_array_equal(first[0], run(lane.readout(self.prompts[0], self.letters)))
        np.testing.assert_array_equal(first[1], run(lane.readout(self.prompts[1], self.letters)))

    def test_requests_can_interleave_without_sharing_mutable_state(self):
        other = [p.replace('a' * 300, 'b' * 300) for p in self.prompts]
        a, b = run(self.be.share_prefix(self.prompts)), run(self.be.share_prefix(other))
        expected = [run(a.readout(self.prompts[0], self.letters)), run(b.readout(other[0], self.letters))]
        async def interleave():
            return await asyncio.gather(a.readout(self.prompts[0], self.letters),
                                        b.readout(other[0], self.letters),
                                        a.readout(self.prompts[0], self.letters))
        got = run(interleave())
        for x, y in zip(got, [expected[0], expected[1], expected[0]]):
            np.testing.assert_array_equal(x, y)

    def test_mismatching_tokens_are_recomputed(self):
        lane = run(self.be.share_prefix(self.prompts))
        different = self.prompts[0].replace('a' * 300, 'c' * 300)
        np.testing.assert_array_equal(run(self.be.readout(different, self.letters)),
                                      run(lane.readout(different, self.letters)))

    def test_disabled_instant_media_and_short_prefix_do_not_prefill(self):
        be = self.be
        be.prefix_cache = False
        self.assertIsNone(run(be.share_prefix(self.prompts)))
        be.prefix_cache = True
        self.assertIsNone(run(be.share_prefix(['abcX', 'abcY'])))
        self.assertIsNone(run(be.share_prefix(self.prompts, media=object())))
        be.bidirectional = True
        try:
            self.assertIsNone(run(be.share_prefix(self.prompts)))
        finally:
            be.bidirectional = False

    def test_token_prefix_not_character_prefix(self):
        old = self.be.tokenizer
        class BoundaryTokenizer:
            def encode(self, s, add_special_tokens=False):
                return [10] * 100 + [11 if s.endswith('route?') else 12] + [13] * 300
        self.be.tokenizer = BoundaryTokenizer()
        try:
            # Text shares >300 characters; token sequences diverge at token 100.
            self.assertIsNone(run(self.be.share_prefix(self.prompts)))
        finally:
            self.be.tokenizer = old

    def test_identical_prompts_leave_a_token_for_readout_and_report_savings(self):
        lane = run(self.be.share_prefix([self.prompts[0]] * 2))
        n = len(self.be.tokenizer.encode(self.prompts[0]))
        self.assertEqual(lane.shared_tokens, n - 1)
        for _ in range(2):
            run(lane.readout(self.prompts[0], self.letters))
        self.assertEqual(lane.cache_stats['saved_tokens'], n - 1)
        self.assertEqual(lane.cache_stats['evaluated_tokens'], n + 1)

if __name__ == '__main__':
    unittest.main()
