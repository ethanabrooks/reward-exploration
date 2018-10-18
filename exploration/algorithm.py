import itertools
import time

import numpy as np

from exploration.gridworld import Gridworld


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    X = np.array(X)

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.max(y, axis=axis, keepdims=True)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.sum(y, axis=axis, keepdims=True)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def model_based_visit(visit_func, env: Gridworld,
                      n_policies: int = 0, alpha: float = .1,
                      gamma: float = .99,
                      delta: float = .001):
    R = np.zeros((env.nS, 1, 1))
    given_R = np.vectorize(visit_func)(np.arange(env.nS)).reshape((env.nS, 1, 1))
    D = np.zeros((env.nS, 1, env.nS))
    for i in range(env.nS):
        D[i, :, i] = 1
    Q = 1e-5 * np.ones((env.nS, env.nA, 1))
    delD = np.zeros((env.nS, 1, env.nS))
    delQ = np.zeros((env.nS, env.nA, 1))
    P = env.transition_matrix
    ER = np.sum(D * given_R)

    def interpolate(x, x_new):
        x += alpha * (x_new - x)

    for i in itertools.count():
        time.sleep(.2)
        pi = softmax(Q, axis=1)
        interpolate(D, np.sum(pi * P * D, axis=(1, 2), keepdims=True))
        for i in range(env.nS):
            D[i, : i] = 1
        dPi_dQ = pi * (np.eye(env.nA) - pi.transpose([0, 2, 1]))
        delPi = np.matmul(dPi_dQ, delQ)

        import ipdb;
        ipdb.set_trace()
        interpolate(delQ, 1 + np.sum(P * (delPi * D + delQ * pi), axis=(1, 2),
                                     keepdims=True))
        interpolate(delD,
                    np.sum(P * (delPi * D + delD * pi), axis=(1, 2), keepdims=True))
        R += np.sum(delD[1] * given_R, axis=2, keepdims=True)
        Q2 = np.sum(P * np.sum(pi * Q, axis=1, keepdims=True).transpose([1, 2, 0]),
                    axis=2, keepdims=True)
        interpolate(Q, R + gamma * Q2)
        new_ER = np.sum(D * given_R)
        print(R.squeeze())
        # print(Q.squeeze())
        print(D.squeeze())
        # print(new_ER)
        # print(new_ER - ER)

        import ipdb;
        ipdb.set_trace()
        if np.abs(new_ER - ER) < delta:
            return R, Q, D
        ER = new_ER


if __name__ == '__main__':
    env = Gridworld(desc=['◻◻◻◻◻◻'], rewards=dict(), terminal=dict())
    R, Q, D = model_based_visit(lambda s: float(s == 5), env=env)
    print('R')
    print(R)
    print('Q')
    print(Q)
    print('D')
    print(D)
