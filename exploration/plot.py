import numpy as np
import plotly
from plotly import graph_objs as go


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
