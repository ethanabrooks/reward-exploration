#! /usr/bin/env python

from gym.envs.toy_text.discrete import DiscreteEnv
import numpy as np
from typing import Dict, Iterable, Tuple, List
from gym import utils
from six import StringIO


class Gridworld(DiscreteEnv):
    def __init__(self,
                 desc: Iterable[Iterable[str]],
                 terminal: Dict[str, bool],
                 rewards: Dict[str, float],
                 actions=np.array([
                     [0, 1],
                     [1, 0],
                     [0, -1],
                     [-1, 0],
                 ]),
                 action_strings: str = "➡️⬇️⬅️⬆️ ",
                 wall_bump_reward: float = 0):
        self.desc = _desc = np.array(
            [list(r) for r in desc])  # type: np.ndarray
        nrows, ncols = _desc.shape

        def transition_tuple(i: int, j: int) -> Tuple[float, int, float, bool]:
            i = np.clip(i, 0, nrows - 1)
            j = np.clip(j, 0, ncols - 1)
            letter = str(_desc[i, j])
            return (
                1.,
                self.encode(*(np.array([i, j]) + action)),
                rewards.get(letter, 0),
                terminal.get(letter, False),
            )

        P = {
            self.encode(i, j): {
                a: [transition_tuple(i, j, action)]
                for a, action in enumerate(actions)
            }
            for i in range(nrows) for j in range(ncols)
        }
        isd = np.ones(_desc.size) / _desc.size
        super().__init__(
            nS=_desc.size,
            nA=len(actions),
            P=P,
            isd=np.array(isd),
        )

    def render(self):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        out = [[c.decode('utf-8') for c in line] for line in self.desc.copy()]
        i, j = self.decode(self.s)
        out[i, j] = utils.colorize(out[i, k], 'yellow', highlight=True)
        for row in out:
            outfile.write("".join(row) + "\n")
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(
                self.action_strings[self.lastaction]))
        else:
            outfile.write("\n")
        # No need to return anything for human
        if mode != 'human':
            return outfile

    def encode(self, i: int, j: int) -> int:
        nrow, ncol = self.desc.shape
        assert 0 <= i < nrow
        assert 0 <= j < ncol
        return i * ncol + j

    def decode(self, s: int) -> Tuple[int, int]:
        nrow, ncol = self.desc.shape
        assert 0 <= s < nrow * ncol
        return s // nrow, s % ncol


if __name__ == '__main__':
    env = Gridworld(
        desc=['_t', '__'],
        rewards=dict(t=1),
        terminal=dict(t=True),
    )
    env.reset()
    while True:
        env.render()
        print('hello')
