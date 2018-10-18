from typing import Dict, List, Tuple, Any
import numpy as np

TransitionTuple = Tuple[float, int, float, bool]


class DiscreteEnv:
    def __init__(
            self,
            nS: int,
            nA: int,
            P: Dict[int, Dict[int, List[TransitionTuple]]],
            isd: np.ndarray,
    ) -> None:
        ...

    def step(self, action: int) -> Tuple[Any, float, bool, dict]:
        ...

    def reset(self) -> int:
        ...
