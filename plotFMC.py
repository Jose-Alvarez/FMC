# Plotting functions for FMC
#
# FMC, Focal Mechanisms Classification
# Copyright (C) 2013  Jose A. Alvarez-Gomez
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# version 1.6
# Diamond source-type diagram included
# Some plot adjustments were done
# version 1.9
# The scale of the symbols now is adjusted to the magnitude range of the input data
# version 1.9.1
# Symbol labels (-pa) working with recent NumPy versions
# Source type diagram working with a single event or equal magnitudes
# plt.axes() replaced by plt.gca()
# Legend of magnitudes formatted as 8.0 (was 8.); 'Text label' no longer clipped in the source type diagram
# version 1.10
# Left label of the source type diagram changed to CLVD (+) (positive CLVD: tension-dominated)
# Text label of the source type diagram kept at its original position (x = 1.7) and moved to the left only when it would be cut by the edge of the figure
# Magnitude legend of the source type diagram: the symbol of maximum magnitude keeps its original position (x = 1.15), as in the classification diagram

import matplotlib.pyplot as plt
from numpy import zeros, sqrt, arcsin, pi, sin, squeeze, round, asarray
from functionsFMC import kave

plt.rc('pdf', fonttype=3)

def baseplot(spacing, plotname):
    # border
    fig = plt.figure()
    plt.gca().set_aspect('equal')

    X = zeros((1, 101))
    Y = zeros((1, 101))
    for a in range(0, 101):
        P = arcsin(sqrt(a / 100.0)) / (pi / 180)
        B = 0.0
        T = arcsin(sqrt(1 - (a / 100.0))) / (pi / 180)
        X[0][a], Y[0][a] = kave(T, B, P)

    tickx, ticky = kave(range(90, -1, -10), zeros((1, 10)), range(0, 91, 10))
    plt.plot(X[0], Y[0], color='black', linewidth=2)
    plt.scatter(tickx, ticky, marker=3, c='black', linewidth=2)
    for i in range(0, 10):
        plt.text(
            tickx[0][i],
            ticky[0][i] - 0.04,
            i * 10,
            fontsize=8,
            verticalalignment='top')
    plt.text(
        0,
        -0.76,
     'P axis plunge',
     fontsize=9,
     horizontalalignment='center')

    X = zeros((1, 101))
    Y = zeros((1, 101))
    for a in range(0, 101):
        B = arcsin(sqrt(a / 100.0)) / (pi / 180)
        P = 0.0
        T = arcsin(sqrt(1 - (a / 100.0))) / (pi / 180)
        X[0][a], Y[0][a] = kave(T, B, P)

    tickx, ticky = kave(zeros((1, 10)), range(0, 91, 10), range(90, -1, -10))
    plt.plot(X[0], Y[0], color='black', linewidth=2)
    plt.scatter(tickx, ticky, marker=0, c='black', linewidth=2)
    for i in range(0, 10):
        plt.text(
            tickx[0][i] - 0.04,
            ticky[0][i],
            i * 10,
            fontsize=8,
            horizontalalignment='right')
    plt.text(
        -0.7,
        0.2,
     'B axis plunge',
     fontsize=9,
     horizontalalignment='center',
     rotation=60)

    X = zeros((1, 101))
    Y = zeros((1, 101))
    for a in range(0, 101):
        T = arcsin(sqrt(a / 100.0)) / (pi / 180)
        B = 0.0
        P = arcsin(sqrt(1 - (a / 100.0))) / (pi / 180)
        X[0][a], Y[0][a] = kave(T, B, P)

    tickx, ticky = kave(range(0, 91, 10), range(90, -1, -10), zeros((1, 10)))
    plt.plot(X[0], Y[0], color='black', linewidth=2)
    plt.scatter(tickx + 0.025, ticky, marker=0, c='black', linewidth=2)
    for i in range(0, 10):
        plt.text(
            tickx[0][i] + 0.04,
            ticky[0][i],
            i * 10,
            fontsize=8,
            horizontalalignment='left')
    plt.text(
        0.7,
        0.2,
     'T axis plunge',
     fontsize=9,
     horizontalalignment='center',
     rotation=-60)

    X = zeros((1, 101))
    Y = zeros((1, 101))
    for a in range(0, 101):
        P = arcsin(sqrt(a / 100.0)) / (pi / 180)
        T = 0.0
        B = arcsin(sqrt(1 - (a / 100.0))) / (pi / 180)
        X[0][a], Y[0][a] = kave(T, B, P)

    plt.plot(X[0], Y[0], color='black', linewidth=2)

    # inner lines
    # class fields
    X = zeros((1, 51))
    Y = zeros((1, 51))
    for a in range(0, 51):
        B = arcsin(sqrt((a / 50.0) * 0.14645)) / (pi / 180)
        T = 67.5
        P = arcsin(sqrt((1 - (a / 50.0)) * 0.14645)) / (pi / 180)
        X[0][a], Y[0][a] = kave(T, B, P)

    xf = X[0][25]
    yf = Y[0][25]
    plt.plot([0, xf], [0, yf], color='grey', linewidth=1)
    plt.plot(X[0][25:51], Y[0][25:51], color='grey', linewidth=1)

    X = zeros((1, 51))
    Y = zeros((1, 51))
    for a in range(0, 51):
        B = arcsin(sqrt((a / 50.0) * 0.14645)) / (pi / 180)
        P = 67.5
        T = arcsin(sqrt((1 - (a / 50.0)) * 0.14645)) / (pi / 180)
        X[0][a], Y[0][a] = kave(T, B, P)

    xf = X[0][25]
    yf = Y[0][25]
    plt.plot([0, xf], [0, yf], color='grey', linewidth=1)
    plt.plot(X[0][25:51], Y[0][25:51], color='grey', linewidth=1)

    X = zeros((1, 51))
    Y = zeros((1, 51))
    for a in range(0, 51):
        T = arcsin(sqrt((a / 50.0) * 0.14645)) / (pi / 180)
        B = 67.5
        P = arcsin(sqrt((1 - (a / 50.0)) * 0.14645)) / (pi / 180)
        X[0][a], Y[0][a] = kave(T, B, P)

    plt.plot(X[0], Y[0], color='grey', linewidth=1)

    plt.plot([0, 0], [0.555221438, -0.605810893], color='grey', linewidth=1)
    plt.plot([0, 0.52481139], [0, 0.303], color='grey', linewidth=1)
    plt.plot([0, -0.52481139], [0, 0.303], color='grey', linewidth=1)
    # Labels
    plt.text(0, -0.9, plotname, horizontalalignment='center', fontsize=16)
    plt.text(-0.9, -0.5, 'Normal', horizontalalignment='right', fontsize=14)
    plt.text(0.9, -0.5, 'Reverse', horizontalalignment='left', fontsize=14)
    plt.text(0, 1, 'Strike-slip', horizontalalignment='center', fontsize=14)

    plt.axis('off')
    if spacing != 0:
        fig = grids(spacing, plotname)

    return fig


def circles(X, Y, size, color, plotname, label, spacing):
    # working on it
    # computes min and max size to provide adecuate symbol scaling
    minval = min(size)
    maxval = max(size)
    minscale=20
    maxscale=140
    if minval == maxval:
    	symb_size = [100 for valor in size]
    else:
    	symb_size = [(minscale + (valor - minval) * (maxscale - minscale) / (maxval - minval)) for valor in size]
        
    fig = baseplot(spacing, plotname)
    if str(color) == 'white':
        sc = plt.scatter(X, Y, s=symb_size, c=color, alpha=0.7, linewidth=0.5, edgecolors='black')
    else:
        sc = plt.scatter(
            X,
            Y,
            s=symb_size,
            c=color, # AQUI HAY UN PROBLEMA AL UTILIZAR NUMEROS EN ID PARA COLOREAR PROBLEMA EN LA FUNCION COLOR DE matplotlib 3
            alpha=0.7,
            linewidth=0.5,
            edgecolors='black',
            cmap='plasma_r')
        cbar = plt.colorbar(sc, shrink=0.5)
        cbar.set_label(label)
    # legend
    plt.scatter(0.4, 0.9, s=20, c='white', linewidth=0.5, edgecolors='black')
    plt.scatter(0.5, 0.9, s=50, c='white', linewidth=0.5, edgecolors='black')
    plt.scatter(0.6, 0.9, s=80, c='white', linewidth=0.5, edgecolors='black')
    plt.scatter(0.7, 0.9, s=110, c='white', linewidth=0.5, edgecolors='black')
    plt.scatter(0.8, 0.9, s=140, c='white', linewidth=0.5, edgecolors='black')
    plt.text(0.4, .95, '%.1f' % float(asarray(minval).ravel()[0]), fontsize=9, ha='center')
    plt.text(0.8, .95, '%.1f' % float(asarray(maxval).ravel()[0]), fontsize=9, ha='center')
    plt.text(0.9, .95, 'Mw', fontsize=9)
    return fig


def annot(X, Y, size, color, plotname, label, annots, lab_param, spacing):

    fig = circles(X, Y, size, color, plotname, label, spacing)
    for i, txt in enumerate(annots):
        plt.annotate(
            str(txt).strip(".'[]'"),
            (float(asarray(X[i]).ravel()[0]) + 0.01,
             float(asarray(Y[i]).ravel()[0]) + 0.01),
            horizontalalignment='left',
            verticalalignment='bottom',
            rotation=30,
            size='x-small')
    plt.text(
        1.05,
        -0.75,
        'Text label:\n' + str(
            lab_param).strip(
             "'[]'").replace(
             "_",
             " "),
             fontsize=10,
             horizontalalignment='center',
             verticalalignment='top')

    return fig


def grids(spacing, plotname):
    for sp in range(0, 91, spacing):
        # B plunge gridlines
        compl = (sin((90 - sp) * (pi / 180)))**2
        X = zeros((1, 51))
        Y = zeros((1, 51))
        for a in range(0, 51):
            P = arcsin(sqrt((a / 50.0) * compl)) / (pi / 180)
            B = sp
            T = arcsin(sqrt((1 - (a / 50.0)) * compl)) / (pi / 180)
            X[0][a], Y[0][a] = kave(T, B, P)
        plt.plot(X[0], Y[0], color='gray', linewidth=0.5, linestyle='--')
    # P plunge gridlines
        compl = (sin((90 - sp) * (pi / 180)))**2
        X = zeros((1, 51))
        Y = zeros((1, 51))
        for a in range(0, 51):
            B = arcsin(sqrt((a / 50.0) * compl)) / (pi / 180)
            P = sp
            T = arcsin(sqrt((1 - (a / 50.0)) * compl)) / (pi / 180)
            X[0][a], Y[0][a] = kave(T, B, P)
        plt.plot(X[0], Y[0], color='gray', linewidth=0.5, linestyle='--')
    # T plunge gridlines
        compl = (sin((90 - sp) * (pi / 180)))**2
        X = zeros((1, 51))
        Y = zeros((1, 51))
        for a in range(0, 51):
            B = arcsin(sqrt((a / 50.0) * compl)) / (pi / 180)
            T = sp
            P = arcsin(sqrt((1 - (a / 50.0)) * compl)) / (pi / 180)
            X[0][a], Y[0][a] = kave(T, B, P)
        plt.plot(X[0], Y[0], color='gray', linewidth=0.5, linestyle='--')

# Source type diagram, diamond skeewed from Hudson et al. (1989)
def diamond_base(plotname):
    fig = plt.figure()
    plt.gca().set_aspect('equal')

    plt.plot([0,1.3333,0,-1.3333,0],[1,0.3333,-1,-0.3333,1],linewidth=2,color='black') # diagram limits
    plt.plot([-1,1],[0,0],linewidth=1,color='black') # horizontal line
    plt.plot([0,0],[-1,1],linewidth=1,color='black') # vertical line
    plt.plot([-1.3333,1.3333],[-0.3333,0.3333],linewidth=1,color='black',linestyle='--') # diagonal line

    plt.text(0, -1.4, plotname, horizontalalignment='center', fontsize=16) # figure title
    # Labels
    plt.text(0,-1.1,'Implosion', horizontalalignment='center', fontsize=10)
    plt.text(0,1.05,'Explosion', horizontalalignment='center', fontsize=10)
    plt.text(1.05,0,'CLVD (-)', horizontalalignment='left', verticalalignment='center', fontsize=10)
    plt.text(-1.05,0,'CLVD (+)', horizontalalignment='right', verticalalignment='center', fontsize=10)

    plt.axis('off')

    return fig

def diamond_circles(u, v, size, color, plotname, label):
# computes min and max size to provide adecuate symbol scaling
    minval = min(size)
    maxval = max(size)
    minscale=20
    maxscale=100
    if minval == maxval:
        symb_size = [60 for valor in size]
    else:
        symb_size = [(minscale + (valor - minval) * (maxscale - minscale) / (maxval - minval)) for valor in size]

    fig = diamond_base(plotname)
    if str(color) == 'white':
        sc = plt.scatter(u, v, s=symb_size, c=color, alpha=0.7, linewidth=0.5, edgecolors='black')
    else:
        sc = plt.scatter(
            u,
            v,
            s=symb_size,
            c=color, # AQUI HAY UN PROBLEMA AL UTILIZAR NUMEROS EN ID PARA COLOREAR PROBLEMA EN LA FUNCION COLOR DE matplotlib 3
            alpha=0.7,
            linewidth=0.5,
            edgecolors='black',
            cmap='plasma_r')
        cbar = plt.colorbar(sc, shrink=0.5)
        cbar.set_label(label)

        # legend: the symbol of maximum magnitude keeps its original position (x = 1.15), as in the
        # classification diagram, where the five symbols occupy the last five positions of the former legend
    plt.scatter(0.55, 0.9, s=20, c='white', linewidth=0.5, edgecolors='black')
    plt.scatter(0.70, 0.9, s=40, c='white', linewidth=0.5, edgecolors='black')
    plt.scatter(0.85, 0.9, s=60, c='white', linewidth=0.5, edgecolors='black')
    plt.scatter(1.00, 0.9, s=80, c='white', linewidth=0.5, edgecolors='black')
    plt.scatter(1.15, 0.9, s=100, c='white', linewidth=0.5, edgecolors='black')
    plt.text(0.55, .97, '%.1f' % float(asarray(minval).ravel()[0]), fontsize=9, ha='center')
    plt.text(1.15, .97, '%.1f' % float(asarray(maxval).ravel()[0]), fontsize=9, ha='center')
    plt.text(1.31, .97, 'Mw', fontsize=9)
    return fig

def label_inside(fig, x, y, txt, **kwargs):
    """Writes a text label in data coordinates at (x, y), the original position of the 'Text label'
    of the source type diagram, and moves it to the left only if it would be cut by the right edge
    of the figure (for instance when there is no colour bar and the diagram is drawn larger)."""
    ax = fig.axes[0]
    t = ax.text(x, y, txt, **kwargs)
    try:
        fig.canvas.draw()    # final position of the axes is needed to know the extent of the text
        bb = t.get_window_extent(renderer=fig.canvas.get_renderer())
        overflow = bb.x1 - 0.985 * fig.bbox.width
        if overflow > 0:
            pix_per_unit = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
            t.set_x(x - overflow / pix_per_unit)
    except Exception:
        pass    # backends without renderer: the label stays at its original position
    return t


def diamond_annot(X, Y, size, color, plotname, label, annots, lab_param):

        fig = diamond_circles(X, Y, size, color, plotname, label)
        for i, txt in enumerate(annots):
            plt.annotate(
                str(txt).strip(".'[]'"),
                (float(asarray(X[i]).ravel()[0]) + 0.01,
                 float(asarray(Y[i]).ravel()[0]) + 0.01),
                horizontalalignment='left',
                verticalalignment='bottom',
                rotation=30,
                size='x-small')
        label_inside(
            fig,
            1.7,
            -1.25,
            'Text label:\n' + str(
                lab_param).strip(
                "'[]'").replace(
                "_",
                " "),
                fontsize=10,
                horizontalalignment='center',
                verticalalignment='top')

        return fig
