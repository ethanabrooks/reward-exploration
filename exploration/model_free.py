import itertools

import numpy as np

from exploration.gridworld import Gridworld
from exploration.replay_buffer import ReplayBuffer
from exploration.util import softmax


def update_Q(P, Q, R, gamma):
    return np.sum(R + gamma * P * np.max(Q, axis=3), axis=2)


def update_delD(D, P, delD, delPi, pi):
    return np.sum(P * (delPi * D + pi * delD), axis=(1, 2))


def update_delQ(I, P, Q, delPi, delQ, pi):
    return np.sum(I + P * (delPi * Q + pi * delQ), axis=(2, 3))


def update_delPi(I, delQ, pi, nS, nA):
    dPi_dQ = (pi * (I - pi.transpose([0, 2, 1]))).reshape(nS, nA, nA, 1)
    return np.sum(dPi_dQ * delQ, axis=2)


def update_D(D, I, P, pi):
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

    t = True
    s1 = env.reset()
    buffer = ReplayBuffer(maxlen=env.nS * 10)

    # train loop
    for i in itertools.count():
        if t:
            s1 = env.reset()

        pi = softmax(Q, axis=1)
        s2, r, t, i = env.step(Q[s1].max(axis=1))
        buffer.append((s1, s2, t))

        # s, a, s', g
        D = update_D(
            D=(D.reshape(1, 1, env.nS, env.nS)),
            I=(I.reshape(env.nS, 1, 1, env.nS)),
            P=(P.reshape(env.nS, env.nA, env.nS, 1)),
            pi=(pi.reshape(env.nS, env.nA, 1, 1)))
        assert not np.any(np.isnan(D))

        # s, a_pi, a_Q
        # s, a_pi, a_Q, s_r
        delPi = update_delPi(
            I=(np.eye(env.nA).reshape(1, env.nA, env.nA)),
            delQ=(delQ.reshape(env.nS, 1, env.nA, env.nS)),
            pi=(pi.reshape(env.nS, env.nA, 1)),
            nS=env.nS,
            nA=env.nA)
        assert delPi.shape == (env.nS, env.nA, env.nS)
        assert not np.any(np.isnan(delPi))

        # s_Q, a, s', a', s_r
        delQ = update_delQ(
            I=(I.reshape(env.nS, 1, 1, 1, env.nS)),
            P=(P.reshape(env.nS, env.nA, env.nS, 1, 1)),
            Q=(Q.reshape(1, 1, env.nS, env.nA, 1)),
            delPi=(delPi.reshape(1, 1, env.nS, env.nA, env.nS)),
            delQ=(delQ.reshape(1, 1, env.nS, env.nA, env.nS)),
            pi=(pi.reshape(1, 1, env.nS, env.nA, 1)))
        assert not np.any(np.isnan(delQ))

        # s_D, a, s', g, s_r
        delD = update_delD(
            D=(D.reshape(1, 1, env.nS, env.nS, 1).copy()),
            P=(P.reshape(env.nS, env.nA, env.nS, 1, 1).copy()),
            delD=(delD.reshape(1, 1, env.nS, env.nS, env.nS).copy()),
            delPi=(delPi.reshape(env.nS, env.nA, 1, 1, env.nS).copy()),
            pi=(pi.reshape(env.nS, env.nA, 1, 1, 1).copy()))
        assert not np.any(np.isnan(delD))

        # s, a, s', (a')
        Q = update_Q(
            P=(P.reshape(env.nS, env.nA, env.nS)),
            Q=(Q.reshape(1, 1, env.nS, env.nA)),
            R=(R.reshape(env.nS, 1, 1)),
            gamma=gamma)
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
