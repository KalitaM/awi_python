import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.colors
import re
import pandas as pd
from utils import read_ttr

def getPhaseData(filename, frequency=None):
    time_interval, data = read_ttr(filename)
    traces, samples = data.shape
    print("traces", traces, "samples", samples)

    frequencies = np.linspace(0, 1 / time_interval, samples)
    index = int(frequency * time_interval * samples)
    print(len(frequencies), index, frequencies[index], frequency)

    pred = data[:traces//2, :]
    actual = data[traces//2:, :]
    fft_pred = np.fft.fft(pred)
    fft_actual = np.fft.fft(actual)

    phase_pred = np.angle(fft_pred)
    phase_actual = np.angle(fft_actual)

    correlations = []
    for i in range(pred.shape[0]):
        pd_predict = pd.Series(pred[i])
        pd_actual = pd.Series(actual[i])
        correlations.append(pd_predict.corr(pd_actual.shift(0)))

    print(len(correlations))
    return phase_pred[:, index], phase_actual[:, index], correlations


def simple_plot(filename, frequency):

    plot_data = getPhaseData(filename, frequency)

    titles = ["Predict", "Actual"]
    fig, axes = plt.subplots(nrows=2, sharex=True)
    for idx, ax in enumerate(axes):
        im = ax.imshow(plot_data[idx][np.newaxis,:], aspect="auto", vmin=-math.pi, vmax=math.pi)
        ax.set_yticks([])
        ax.title.set_text(titles[idx] + " with f = " + str(frequency))
    
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    plt.show()


def spatial_plot(filename, frequency):
    pre, actual, corr = getPhaseData(filename, frequency)
    # print("Finish getting phase plot data")
    titles = ["Predict", "Actual"]
    rcvr_file = 'rcvrlist' + filename[7:-3] + 'txt'
    rcvr_data = np.loadtxt(rcvr_file, dtype=np.int32, skiprows=1)
    sourceRef = int(re.findall(r"csref(\d+)", filename)[0])
    print("sourceRef is " + str(sourceRef))
    sourceXs = [9924.545]
    sourceYs = [5837.576]

    groupXs = (rcvr_data[:, 4] - 25) * 50
    groupYs = (rcvr_data[:, 3] - 25) * 50
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])
    for i, d in enumerate([pre, actual]):
        plt.figure()
        sc = plt.scatter(groupXs, groupYs, c=d, cmap=cmap, edgecolor='none', s=1)   
        plt.colorbar(sc)
        plt.scatter(sourceXs, sourceYs, s=100, marker='+', c='black') # source scatter
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.title(titles[i] + ' phase plot for frequency ' + str(frequency))
    
    plt.figure()
    sc = plt.scatter(groupXs, groupYs, c=corr, cmap=cmap, edgecolor='none', s=1)   
    plt.colorbar(sc)
    plt.scatter(sourceXs, sourceYs, s=100, marker='+', c='black') # source scatter
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title('correlation plot')
    plt.show()

filename = 'compare-csref01004-iter00012fwd2.ttr'
# filename = "compare-csref02398-iter00029fwd2.ttr"
# simple_plot(filename, 2)
# spatial_plot(filename, 3)



def get_phase_data_multiple(filename, fs):
    time_interval, data = read_ttr(filename)
    traces, samples = data.shape
    print("traces", traces, "samples", samples)

    frequencies = np.linspace(0, 1 / time_interval, samples)
    indice = [int(frequency * time_interval * samples) for frequency in fs]

    pred = data[:traces//2, :]
    actual = data[traces//2:, :]
    fft_pred = np.fft.fft(pred)
    fft_actual = np.fft.fft(actual)

    phase_pred = np.angle(fft_pred)
    phase_actual = np.angle(fft_actual)


    correlations = []
    for i in range(pred.shape[0]):
        pd_predict = pd.Series(pred[i])
        pd_actual = pd.Series(actual[i])
        correlations.append(pd_predict.corr(pd_actual.shift(0)))

    return phase_pred[:, indice], phase_actual[:, indice], correlations



def spatial_plot_multiple(filename, fs):
    pred, actual, corr = get_phase_data_multiple(filename, fs)
    print("Finish getting phase plot data")

    titles = ["Predict", "Actual"]
    rcvr_file = 'rcvrlist' + filename[7:-3] + 'txt'
    rcvr_data = np.loadtxt(rcvr_file, dtype=np.int32, skiprows=1)
    sourceRef = int(re.findall(r"csref(\d+)", filename)[0])
    print("sourceRef is " + str(sourceRef))
    sourceXs = [9924.545]
    sourceYs = [5837.576]

    groupXs = (rcvr_data[:, 4] - 25) * 50
    groupYs = (rcvr_data[:, 3] - 25) * 50
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])

    for freq in range(pred.shape[1]):
        for i, d in enumerate([pred[:, freq], actual[:, freq]]):
            plt.figure()
            sc = plt.scatter(groupXs, groupYs, c=d, cmap=cmap, edgecolor='none', s=1)   
            plt.colorbar(sc)
            plt.scatter(sourceXs, sourceYs, s=100, marker='+', c='black') # source scatter
            plt.xlabel("$x$")
            plt.ylabel("$y$")
            plt.title(titles[i] + ' phase plot for frequency ' + str(fs[freq]))

    plt.figure()
    sc = plt.scatter(groupXs, groupYs, c=corr, cmap=cmap, edgecolor='none', s=1)   
    plt.colorbar(sc)
    plt.scatter(sourceXs, sourceYs, s=100, marker='+', c='black') # source scatter
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.title('correlation plot')
    plt.show()


spatial_plot_multiple(filename, [2,3])
