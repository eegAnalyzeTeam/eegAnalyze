from matplotlib import pyplot as plt
import pandas as pd
import numpy as np


def get_pvalue():
    N = 62
    map_pvalue = pd.read_csv('coh_anova_pvalue.csv')

    return  map_pvalue,N

def draw(data, xlabels, ylabels,name,isgary):
    plt.style.use('classic')
    figure = plt.figure(facecolor='lightgrey')
    ax = figure.add_subplot(111)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    if  isgary:
        _map = ax.imshow(data, interpolation='nearest', cmap=plt.cm.gray, aspect='auto')
    else:
        _map = ax.imshow(data, interpolation='nearest', cmap=plt.cm.rainbow, aspect='auto', vmin=0, vmax=0.02)
        plt.colorbar(mappable=_map)
    plt.title(name)
    plt.savefig(name)
    plt.close()

def draw_p_value():
    map_pvalue,N = get_pvalue()

    xlabels = map_pvalue.columns.tolist()
    ylabels = map_pvalue.columns.tolist()[::-1]
    print(xlabels, ylabels)

    lists = []
    for i in range(N):
        print(i)
        lists.append(map_pvalue.loc[i][0:N].tolist())

    map_pvalue = np.array(lists)

    draw(map_pvalue,xlabels,ylabels,'anova_pvalue',False)

def check_pvalue(x):
    if x<0.01:
        return 1
    else:
        return 0

def draw_p_value_gray():
    map_pvalue,N = get_pvalue()

    xlabels = map_pvalue.columns.tolist()
    ylabels = map_pvalue.columns.tolist()[::-1]
    print(xlabels, ylabels)

    lists = []
    for i in range(N):
        print(i)
        lists.append(list(map(check_pvalue, map_pvalue.loc[i][0:N])))

    map_pvalue = np.array(lists)

    draw(map_pvalue, xlabels, ylabels, 'anova_pvalue_gray',True)

# def draw_histogram():
#     histogram_list = pd.read_csv('coh_anova_histogram.csv')


def anova_plot():
    draw_p_value()
    draw_p_value_gray()

anova_plot()

