import itertools
import time

import numpy as np

from exploration.gridworld import Gridworld
from exploration.util import softmax


def update_Q(P: np.ndarray, Q: np.ndarray, R: np.ndarray, gamma: float):
    return np.sum(R + gamma * P * np.max(Q, axis=3), axis=2)


def update_delD(D: np.ndarray, P: np.ndarray, delD: np.ndarray, delPi: np.ndarray,
                pi: np.ndarray):
    return np.sum(P * (delPi * D + pi * delD), axis=(1, 2))


def update_delQ(I: np.ndarray, P: np.ndarray, Q: np.ndarray, delPi: np.ndarray,
                delQ: np.ndarray, pi: np.ndarray):
    return np.sum(I + P * (delPi * Q + pi * delQ), axis=(2, 3))


def update_delPi(I: np.ndarray, delQ: np.ndarray, pi: np.ndarray, nS: np.ndarray,
                 nA: np.ndarray):
    dPi_dQ = (pi * (I - pi.transpose([0, 2, 1]))).reshape(
        nS, nA, nA, 1)
    return np.sum(dPi_dQ * delQ, axis=2)


def update_D(D: np.ndarray, I: np.ndarray, P: np.ndarray, pi: np.ndarray):
    return np.sum(I + pi * P * D, axis=(1, 2))


def optimize_reward(visit_func,
                    env: Gridworld,
                    alpha: float = .001,
                    gamma: float = .99,
                    delta: float = .001):
    # rewards
    R = np.zeros(env.nS)
    given_R = np.vectorize(visit_func)(np.arange(env.nS))

    # visitation frequency
    D = np.eye(env.nS)
    delD = np.zeros((env.nS, env.nS, env.nS))

    # Q
    Q = 1e-5 * np.ones((env.nS, env.nA))
    I_Q = np.zeros((env.nS, env.nA, env.nS))
    for i in range(env.nS):
        I_Q[i, :, i] = 1
    delQ = np.zeros((env.nS, env.nA, env.nS))

    P = env.transition_matrix
    I = np.eye(env.nS)
    ER = np.sum(D * given_R)
    start_states, = np.nonzero(env.isd)

    # train loop
    for i in itertools.count():
        time.sleep(.2)
        pi = softmax(Q, axis=1)

        # s, a, s', g
        D = update_D(D=(D.reshape(1, 1, env.nS, env.nS)),
                     I=(I.reshape(env.nS, 1, 1, env.nS)),
                     P=(P.reshape(env.nS, env.nA, env.nS, 1)),
                     pi=(pi.reshape(env.nS, env.nA, 1, 1)))
        assert not np.any(np.isnan(D))

        # s, a_pi, a_Q
        _pi = pi.reshape(env.nS, env.nA, 1)
        _I = np.eye(env.nA).reshape(1, env.nA, env.nA)
        dPi_dQ = _pi * (_I - _pi.transpose([0, 2, 1]))
        assert dPi_dQ.shape == (env.nS, env.nA, env.nA)
        assert not np.any(np.isnan(dPi_dQ))

        # s, a_pi, a_Q, s_r
        _delQ = delQ.reshape(env.nS, 1, env.nA, env.nS)
        dPi_dQ = dPi_dQ.reshape(env.nS, env.nA, env.nA, 1)
        delPi = np.sum(dPi_dQ * _delQ, axis=2)
        assert delPi.shape == (env.nS, env.nA, env.nS)
        assert not np.any(np.isnan(delPi))

        # s_Q, a, s', a', s_r
        _I = I.reshape(env.nS, 1, 1, 1, env.nS)
        _P = P.reshape(env.nS, env.nA, env.nS, 1, 1)
        _delPi = delPi.reshape(1, 1, env.nS, env.nA, env.nS)
        _Q = Q.reshape(1, 1, env.nS, env.nA, 1)
        _pi = pi.reshape(1, 1, env.nS, env.nA, 1)
        _delQ = delQ.reshape(1, 1, env.nS, env.nA, env.nS)

        delQ = np.sum(_I + _P * (_delPi * _Q + _pi * _delQ), axis=(2, 3))
        assert not np.any(np.isnan(delQ))

        # s_D, a, s', g, s_r
        _P = P.reshape(env.nS, env.nA, env.nS, 1, 1).copy()
        _delPi = delPi.reshape(env.nS, env.nA, 1, 1, env.nS).copy()
        _D = D.reshape(1, 1, env.nS, env.nS, 1).copy()
        _pi = pi.reshape(env.nS, env.nA, 1, 1, 1).copy()
        _delD = delD.reshape(1, 1, env.nS, env.nS, env.nS).copy()

        delD_before = delD.copy()
        delD = np.sum(_P * (_delPi * _D + _pi * _delD), axis=(1, 2))
        assert not np.any(np.isnan(delD))

        # s, a, s', (a')
        _R = R.reshape(env.nS, 1, 1)
        _P = P.reshape(env.nS, env.nA, env.nS)
        _Q = Q.reshape(1, 1, env.nS, env.nA)
        Q = np.sum(_R + gamma * _P * np.max(_Q, axis=3), axis=2)
        assert not np.any(np.isnan(Q))

        # s0, g, s_r
        _given_R = given_R.reshape(1, env.nS, 1)
        delR = np.sum(delD[start_states] * _given_R, axis=(0, 1))
        R += alpha * delR

        new_ER = np.sum(D * given_R)
        _R = np.round(R * 4)
        if np.any(R):
            a_chars = np.array(tuple(env.action_strings))
            policy_string = a_chars[np.argmax(pi,
                                              axis=1).reshape(env.desc.shape)]
            for r in policy_string:
                print(''.join(r))

            # for i, r in enumerate(_R):
            # print(i, '|' * int(r))
            # print(Q.squeeze())
            # print(D.squeeze())
            # print('new', new_ER)
            # print('diff', new_ER - ER)
            print()

        if np.abs(new_ER - ER) < delta:
            return R, Q, D
        ER = new_ER
    raise RuntimeError
