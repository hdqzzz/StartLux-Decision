import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock

from startlux_decide.backends.base import BackendRequestError
from startlux_decide.pipeline import Decider, _Request


class ContextGuardTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.decider = Decider.__new__(Decider)
        self.decider.tok = lambda text, **kw: {'input_ids': list(text)}
        self.decider.context_length = 8
        self.decider.letter_ids = [65, 66]
        self.decider.folder = SimpleNamespace(temperature=lambda typ: 1.0)
        self.lane = SimpleNamespace(name="mlx", readout=AsyncMock(return_value=[0.0, 0.0]))

    async def test_limit_is_inclusive(self):
        probs = await self.decider._read(self.lane, 'a' * 8, 2, 'choice', _Request())
        self.assertEqual(probs, [0.5, 0.5])
        self.lane.readout.assert_awaited_once()

    async def test_over_limit_does_not_reach_engine(self):
        with self.assertRaises(BackendRequestError):
            await self.decider._read(self.lane, 'a' * 9, 2, 'choice', _Request())
        self.lane.readout.assert_not_awaited()

    async def test_media_tokens_count_towards_limit(self):
        with self.assertRaises(BackendRequestError):
            await self.decider._read(self.lane, 'a' * 7, 2, 'choice', _Request(extra_tokens=2))
        self.lane.readout.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
