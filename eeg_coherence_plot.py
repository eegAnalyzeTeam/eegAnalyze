import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

plt.style.use('classic')

# N = 62
# a = pd.read_csv('eeg_coherence.csv')
#
# xlabels = a.columns.tolist()
# ylabels = list(range(N))
#
# lists = []
# for i in range(N):
#     print(i)
#     lists.append(a.loc[i][0:N].tolist())
#
# a = np.array(lists)


def draw_heatmap(data, xlabels, ylabels,name):
    plt.style.use('classic')
    figure = plt.figure(facecolor='lightgrey')
    ax = figure.add_subplot(111)
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(len(xlabels)))
    ax.set_xticklabels(xlabels)
    map = ax.imshow(data, interpolation='nearest', cmap=plt.cm.rainbow, aspect='auto', vmin=0, vmax=1)
    plt.colorbar(mappable=map)
    if name =='c_coherence':
        plt.title('control')
    else:
        plt.title('patient')
    plt.savefig(name)
    plt.close()


# draw_heatmap(a, xlabels, ylabels)

def coh_plot():
    N = 62
    file =['eeg_coherence_c.csv','eeg_coherence_p.csv']
    count=0
    for csv in file:
        a = pd.read_csv(csv)

        xlabels = a.columns.tolist()
        ylabels = a.columns.tolist()[::-1]
        print(xlabels,ylabels)

        lists = []
        for i in range(N):
            print(i)
            lists.append(a.loc[i][0:N].tolist())

        a = np.array(lists)
        if count==0:
            draw_heatmap(a, xlabels, ylabels,'c_coherence')
        else:
            draw_heatmap(a, xlabels, ylabels, 'p_coherence')
        count+=1
