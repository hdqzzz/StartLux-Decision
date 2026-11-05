"""Prefix reuse stays request-local and does not bypass refusal or thinking paths."""
import unittest
from startlux_decide.backends.fake import FakeBackend
from startlux_decide.controls import Controls, RequestError
from startlux_decide.pipeline import Decider
from tests.test_pipeline import AUTO, QUESTIONS, run


class PrefixBackend(FakeBackend):
    name = "mlx"
    prefix_cache = True

    def __init__(self):
        super().__init__()
        self.shared = 0
        self.shared_tokens = 300

    async def share_prefix(self, prompts, **kwargs):
        self.shared += 1
        return self


class PrefixPipelineTest(unittest.TestCase):
    def test_never_shares_but_thinking_does_not(self):
        be = PrefixBackend()
        d = Decider(AUTO, be, context_length=8192)
        run(d.decide("evidence " * 40, QUESTIONS, Controls(think_mode="never")))
        self.assertEqual(be.shared, 1)
        run(d.decide("evidence " * 40, QUESTIONS, Controls(think_mode="auto", gate=0)))
        self.assertEqual(be.shared, 1)

    def test_oversized_input_is_not_prefilled(self):
        be = PrefixBackend()
        d = Decider(AUTO, be, context_length=8)
        with self.assertRaises(RequestError):
            run(d.decide("evidence " * 40, QUESTIONS, Controls(think_mode="never")))
        self.assertEqual(be.shared, 0)
        self.assertFalse(any(c[0] == "readout" for c in be.calls))

    def test_none_prefix_lane_falls_back_to_the_backend(self):
        class NoPrefix(PrefixBackend):
            async def share_prefix(self, prompts, **kwargs):
                return None
        a = run(Decider(AUTO, NoPrefix()).decide("evidence", QUESTIONS, Controls(think_mode="never")))
        b = run(Decider(AUTO, FakeBackend()).decide("evidence", QUESTIONS, Controls(think_mode="never")))
        self.assertEqual(a.answers, b.answers)
