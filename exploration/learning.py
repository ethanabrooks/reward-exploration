from argparse import ArgumentParser
from collections import namedtuple
from typing import Dict, List, Tuple

import numpy as np

Transition = namedtuple('Transition', 'probability new_state reward')
TransitionDict = Dict[int, Dict[int, List[Transition]]]


def in_bounds(i: int, j: int):
    coord = np.array([i, j])
    return np.all(
        np.logical_and(
            np.zeros_like(coord) <= coord, coord < np.array(DESC.shape)))


def one_hot(indices: np.ndarray, depth: int):
    assert np.max(indices) < depth
    array = np.zeros(indices.shape + (depth, ))
    array[indices] = 1
    return array

def ismax(X: np.array, axis: int):
    array = np.zeros_like(X)
    array[X == X.max(axis=axis, keepdims=True)] = 1
    return array


def get_transition(state: int,
                   action: int,
                   slip_prob: float,
                   terminal_states: str = '') -> List[Transition]:


    def state_to_coord(state: int):
        h, w = DESC.shape
        return np.array([state // w, state % w])


    def coord_to_state(i: int, j: int):
        h, w = DESC.shape
        return i * w + j
        coord = state_to_coord(state)
        new_coord = coord + ACTIONS[action]
        new_state = coord_to_state(*new_coord)
        if DESC[tuple(coord)] in terminal_states:
            return [Transition(probability=1, new_state=state, reward=0)]
        if DESC[tuple(coord)] == 'A':
            new_coord = np.argwhere(DESC == 'Ȧ')[0]
            return [
                Transition(
                    probability=1, new_state=coord_to_state(*new_coord), reward=10)
            ]

    if DESC[tuple(coord)] == 'B':
        new_coord = np.argwhere(DESC == 'Ḃ')[0]
        return [
            Transition(
                probability=1, new_state=coord_to_state(*new_coord), reward=5)
        ]
    if not in_bounds(*new_coord):
        return [Transition(probability=1, new_state=state, reward=-1)]
    return [
        Transition(probability=slip_prob, new_state=state, reward=0),
        Transition(probability=1 - slip_prob, new_state=new_state, reward=0)
    ]


def gridworld(slip_prob: float = 0.2,
              terminal_states: str = '') -> TransitionDict:
    """
    P={
                   s1: {a1: [(p(s'_1|s1,a1), s'_1, reward(s'_1,s1,a1)),
                             (p(s'_2|s1,a1), s'_2, reward(s'_2,s1,a1)),
                             ...
    ], a2: ...,
    ...
    }, s2: ...,
    ... }
    """
    states = list(range(DESC.size))
    return {
        state: {
            action: get_transition(
                state=state,
                action=action,
                slip_prob=slip_prob,
                terminal_states=terminal_states)
            for action in range(len(ACTIONS))
        }
        for state in states
    }


def gridworld_ep(slip_prob=0.2):
    return gridworld(slip_prob=slip_prob, terminal_states='ȦḂ')


def get_matrices(P: TransitionDict) -> Tuple[np.ndarray, np.ndarray]:
    transition_matrix = np.zeros((len(P), len(ACTIONS), len(P)))
    reward_matrix = np.zeros((len(P), len(ACTIONS), len(P)))
    for state, actions in P.items():
        for action, transitions in actions.items():
            for transition in transitions:
                i = state, action, transition.new_state
                transition_matrix[i] = transition.probability
                reward_matrix[i] = transition.reward
    assert np.min(transition_matrix) >= 0
    assert np.max(transition_matrix) <= 1
    assert np.all(np.sum(transition_matrix, axis=2) == 1)
    return transition_matrix, reward_matrix


def _policy_eval(T: np.ndarray, R: np.ndarray, policy: np.ndarray,
                 theta: float, gamma: float) -> np.ndarray:
    policy = policy.reshape(DESC.size, len(ACTIONS), 1)

    def __policy_eval(V: np.ndarray):
        assert T.shape == R.shape == (DESC.size, len(ACTIONS), DESC.size)
        assert policy.shape == (DESC.size, len(ACTIONS), 1)
        assert V.shape == (DESC.size, )

        V = V.reshape(1, 1, DESC.size)  # transpose for S'
        new_V = np.sum(np.sum(policy * T * (R + gamma * V), axis=2), axis=1)
        if np.max(np.abs(new_V - V)) < theta:
            return new_V
        return __policy_eval(new_V)

    return __policy_eval(V=np.zeros((DESC.size, )))


def policy_eval(P: TransitionDict,
                policy: np.ndarray = UNIFORM_POLICY,
                theta=0.0001,
                gamma=0.9,
                latex_format=False) -> str:
    """
    P: as returned by your gridworld(slip=0.2).
    policy: probability distribution over actions for each states.
      Default to uniform policy.
    theta: stopping condition.
    gamma: the discount factor.
    V: 5 by 5 numpy array where each entry is the value of the
      corresponding location. Initialize V with zeros.
    """
    return format_values(
        _policy_eval(
            *get_matrices(P), policy=policy, theta=theta, gamma=gamma),
        latex_format=latex_format)


def _policy_iter(T: np.ndarray, R: np.ndarray, theta=0.0001,
                 gamma=0.9) -> Tuple[np.ndarray, np.ndarray]:
    def __policy_iter(policy: np.ndarray):
        assert policy.shape == (DESC.size, len(ACTIONS))
        V = _policy_eval(policy=policy, T=T, R=R, theta=theta, gamma=gamma)
        assert V.shape == (DESC.size, )
        new_policy = ismax(np.sum(T * (R + gamma * V), axis=2), axis=1)
        new_policy /= np.sum(new_policy, axis=1, keepdims=True)
        if np.max(np.abs(new_policy - policy)) < theta:
            return V, new_policy
        return __policy_iter(policy=new_policy)

    values, policy = __policy_iter(UNIFORM_POLICY)
    assert values.shape == (DESC.size, )
    assert policy.shape == (DESC.size, len(ACTIONS))
    return values, policy


def format_policy(policy: np.ndarray, latex_format: bool) -> str:
    letters = 'NESW' if latex_format else '↑→↓←'
    assert policy.shape == (DESC.size, len(ACTIONS))
    # for r in ismax(policy, axis=1):
        # for a, m in zip(letters, r):
            # print(a, m)
    policy_letters = np.array([''.join([a for a, m in zip(letters, r) if m])
                      for r in ismax(policy, axis=1)])
    policy_letters = np.reshape(policy_letters, DESC.shape)
    if latex_format:
        return '\\\\ \n'.join([' & '.join(r) for r in policy_letters])
    return '\n'.join(['\t'.join(r) for r in policy_letters])


def format_values(values: np.array, latex_format: bool) -> str:
    values = values.reshape(DESC.shape)
    if latex_format:
        return '\\begin{bmatrix}\n' + \
               '\\\\ \n'.join([' & '.join(['{:0.2f}'.format(s) for s in r])
                               for r in values]) + \
               '\n\\end{bmatrix}'
    return '\n'.join([' '.join(['{:0.2f}'.format(s) for s in r]) for r in values])


def policy_iter(P: TransitionDict, theta=0.0001, gamma=0.9, latex_format=False):
    """
    policy: 25 by 4 numpy array where each row is a probability
    distribution over moves for a state. If it is
    deterministic, then the probability will be a one hot vector.
    If there is a tie between two actions, break the tie with
    equal probabilities.
    Initialize the policy with the uniformly random policy
    described in Part (b).
    """
    values, policy = _policy_iter(*get_matrices(P), theta=theta, gamma=gamma)
    return (format_values(values, latex_format=latex_format),
            format_policy(policy, latex_format=latex_format))


def _value_iter(T: np.ndarray, R: np.ndarray, theta: float, gamma: float):
    def __value_iter(V: np.ndarray):
        assert V.shape == (DESC.size, )
        V = V.reshape(1, 1, DESC.size)  # transpose for S'
        new_V = np.max(np.sum(T * (R + gamma * V), axis=2), axis=1)
        if np.max(np.abs(new_V - V)) < theta:
            return new_V
        return __value_iter(V=new_V)

    values = __value_iter(np.zeros(DESC.size))
    policy = ismax(np.sum(T * (R + gamma * values), axis=2), axis=1)
    policy /= np.sum(policy, axis=1, keepdims=True)
    return values, policy


def value_iter(P: TransitionDict, theta=0.0001, gamma=0.9, latex_format=False):
    values, policy = _value_iter(*get_matrices(P), theta=theta, gamma=gamma)
    assert values.shape == (DESC.size, )
    assert policy.shape == (DESC.size, len(ACTIONS))
    return (format_values(values, latex_format=latex_format),
            format_policy(policy, latex_format=latex_format))


def main(method: str, slip_prob: float, theta: float, gamma: float,
         terminal_states: str, latex_format: bool):
    kwargs = dict(
        P=gridworld(slip_prob=slip_prob, terminal_states=terminal_states),
        theta=theta,
        gamma=gamma,
        latex_format=latex_format
    )
    for row in DESC:
        print(''.join(row))
    if method in ('gridworld', 'a'):
        print(gridworld(slip_prob=slip_prob))
    if method in ('policy_eval', 'b'):
        print(policy_eval(**kwargs))
    if method in ('policy_iter', 'c'):
        print(*policy_iter(**kwargs), sep='\n')
    if method in ('value_iter', 'd'):
        print(*value_iter(**kwargs), sep='\n')
    if method in ('gridworld_ep', 'e'):
        print(*gridworld_ep(slip_prob=slip_prob))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        'method',
        choices=[
            'a',
            'gridworld',
            'b',
            'policy_eval',
            'c',
            'policy_iter',
            'd',
            'value_iter',
            'e',
            'gridworld_ep',
        ],
        help='Choose either the name of the function to be run or the letter of'
        'the corresponding programming problem. For example, "b" would select'
        'the "policy_eval" function as would the string "policy_eval".')
    parser.add_argument('--slip-prob', default=.2, type=float)
    parser.add_argument('--theta', default=.0001, type=float)
    parser.add_argument('--gamma', default=.9, type=float)
    parser.add_argument('--terminal-states', default='')
    parser.add_argument('--latex-format', action='store_true')
    args = parser.parse_args()
    main(**vars(args))
