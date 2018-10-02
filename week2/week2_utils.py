"""
This utility file is used to display in a visual pleasant way the Frozen Lake
grid world.
Credit goes to kk1694 (https://krisztiankovacs.com/).
Code source: https://github.com/kk1694/rl_course/blob/master/utils_fl.py
"""
import numpy as np
import gym
import matplotlib.pyplot as plt
import seaborn as sns


def env2img(env, labels=None, char_lab=True, ax=None, lab_col='black'):
    desc = [[c.decode('utf-8') for c in line]
            for line in env.env.desc.tolist()]
    dim = len(desc)
    X = np.zeros((dim, dim))

    if labels is None:
        labels = np.array([list(' '*dim) for i in range(dim)])
    elif labels is 'start_pos':
        labels = np.array([list(' '*dim) for i in range(dim)])
        labels[0, 0] = 'X'
        lab_col = 'red'

    if not char_lab:
        labels = labels.round(2).astype(str)

    for i in range(dim):
        for j in range(dim):
            if desc[i][j] == 'H':
                X[i, j] = -10
                labels[i, j] = ""
            elif desc[i][j] == 'G':
                X[i, j] = 10
                labels[i, j] = ""

    sns.heatmap(X, xticklabels=False, yticklabels=False, linewidths=1, ax=ax,
                annot=labels, fmt="",
                annot_kws={'color': lab_col, 'size': '20'},
                cbar=False, cmap=sns.xkcd_palette(['black', 'blue', 'orange']))
    return ax


def convV2mat(V, env):
    return V.reshape((env.env.ncol, env.env.ncol))


def plotV(V, env):
    """Use this function to plot the grid world environment with values for each
    state (cell).

    Args:
        V (array): values array.
        env (object): OpenAI Gym environment.
    """
    env2img(env, labels=convV2mat(V, env), char_lab=False)


def plotPath(path, env):
    """Use this function to plot the grid with the path (action) for each state
    (cell).
    
    Args:
        path (array): path array.
        env (object): OpenAI Gym environment.
    """
    mapping = {0: '←', 1: '↓', 2: '→', 3: '↑'}
    path = np.array([mapping[i] for i in path])
    env2img(env, labels=convV2mat(path, env), char_lab=True)


def plotPathAndV(path, V, env):
    f = plt.figure(figsize=(10, 5))
    ax = f.add_subplot(121)
    plotPath(path, env)
    ax2 = f.add_subplot(122)
    plotV(V, env)
