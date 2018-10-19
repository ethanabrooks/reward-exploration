"""
the algorithm
"""

# stdlib
import itertools
import time

# third party
import numpy as np
import plotly
from plotly import graph_objs as go

# first party
from exploration.gridworld import Gridworld


def plot_values(env, D, delPi, delQ, delD):
    start_states, = np.nonzero(env.isd)

    def layout(
            title,
            xaxis_title,
            yaxis_title,
            xticktext=None,
            yticktext=None,
    ):
        xtick = dict()
        ytick = dict()
        if yticktext is not None:
            ytick = dict(
                ticktext=yticktext, tickvals=tuple(range(len(yticktext))))
        if xticktext is not None:
            xtick = dict(
                ticktext=xticktext, tickvals=tuple(range(len(xticktext))))
        return dict(
            title=title,
            **{
                f'xaxis{i+1}': dict(title=xaxis_title, **xtick)
                for i in range(env.nS)
            },
            **{
                f'yaxis{i+1}': dict(title=yaxis_title, **ytick)
                for i in range(env.nS)
            },
        )

    xticktext = list("▲S▼")
    plot(D, layout=dict(title='D'))
    plot(
        delPi.transpose([2, 0, 1]),
        layout=layout(
            title='delPi',
            yaxis_title='d pi(a|.)',
            xaxis_title='d pi(.|s)',
            xticktext=xticktext,
        ))
    plot(
        delQ.transpose([2, 0, 1]),
        layout=layout(
            title='delQ',
            yaxis_title='d Q(., a)',
            xaxis_title='d Q(s, .)',
            xticktext=xticktext,
        ))
    plot(
        delD.transpose([2, 0, 1]),
        layout=layout(
            title='delD',
            yaxis_title='d D(., g)',
            xaxis_title='d D(s0, .)',
        ))
    plot(delD[start_states, 5, :], layout=dict(title='delR'))
    import ipdb
    ipdb.set_trace()


def plot(X: np.ndarray, layout=None, filename='values.html') -> None:
    if layout is None:
        layout = dict()
    *lead_dims, _, _ = X.shape
    if len(lead_dims) == 1:
        lead_dims = [1] + lead_dims

    def iterate_x():
        if len(X.shape) == 2:
            yield 1, 1, X
        elif len(X.shape) == 3:
            for i, matrix in enumerate(X, start=1):
                yield 1, i, matrix
        elif len(X.shape) == 4:
            for i, _tensor in enumerate(X, start=1):
                for j, matrix in enumerate(_tensor, start=1):
                    yield j, i, matrix
        else:
            raise RuntimeError

    fig = plotly.tools.make_subplots(
        *lead_dims, subplot_titles=[str(j - 1) for _, j, _ in iterate_x()])

    for i, j, matrix in iterate_x():
        trace = go.Heatmap(z=matrix, colorscale='Viridis')
        # z=np.flip(matrix, axis=(0, 1)), colorscale='Viridis')
        fig.append_trace(trace, i, j)

    fig['layout'].update(**layout)
    plotly.offline.plot(fig, auto_open=True, filename=filename)


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
    if len(X.shape) == 1:
        p = p.flatten()

    return p


def model_based_visit(visit_func,
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
        _I = I.reshape(env.nS, 1, 1, env.nS)
        _pi = pi.reshape(env.nS, env.nA, 1, 1)
        _P = P.reshape(env.nS, env.nA, env.nS, 1)
        _D = D.reshape(1, 1, env.nS, env.nS)
        D = np.sum(_I + _pi * _P * _D, axis=(1, 2))
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


if __name__ == '__main__':
    ENV = Gridworld(
        desc=[
            '◻◻◻◻◻',
            '◻◻◻◻◻',
            '◻◻◻◻◻',
            '◻◻◻◻◻',
            '◻◻◻◻S',
        ],
        rewards=dict(),
        terminal='T',
        start_states='S',
    )
    # actions=np.array([[0, 1], [0, 0], [0, -1]]),
    # action_strings="▶s◀")
    _R, _Q, _D = model_based_visit(
        lambda s: float(s == 0),
        env=ENV,
    )
    print('_R')
    print(_R)
    print('_Q')
    print(_Q)
    print('_D')
    print(_D)
