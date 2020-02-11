from scipy.linalg import toeplitz, solve_toeplitz
import scipy
from scipy import signal
import numpy as np
from numpy.linalg import solve, norm
from utils import *
import matplotlib.pyplot as plt
import matplotlib.colors

def filter_plot(filename, stabilisation, scale, wtype, wmin, down_sampling=2, verbose=False):
    time_interval, data = read_ttr_compare(filename)
    traces, samples = data.shape

    p = data[:traces//2, :]
    d = data[traces//2:, :]

    print(p.shape)

    # weight
    T = apply_weight(2 * len(d[0]) // down_sampling - 1, scale, wtype, wmin) 


    filters = []
    f_values = []

    for trace in range(p.shape[0]):
        # if trace % 15 == 0:
        #     print(trace)

        # downsampling
        trace_p = list(p[trace, ::down_sampling])
        trace_d = list(d[trace, ::down_sampling])

        # identity filter
        nsamples = len(trace_p)
        filterI = np.zeros([nsamples],dtype=np.float32)
        filterI[int(nsamples/2)] = 1.0
        
        # padded with zeros
        d_pad = scipy.signal.convolve(trace_d,filterI)
        p_pad = scipy.signal.convolve(trace_p,filterI)

        # The goal is to solve  (D^T D + a d^t d I) v = D^T p + a d^T d z;   z = [0..1..0] 2n-1, a = stablisation

        length = d_pad.shape[0]

        # autocorrelation is d^T d
        autocorr = scipy.signal.correlate(trace_d,trace_d) 
        
        # crosscorrelation is D^T p
        xcorr = scipy.signal.correlate(trace_p, trace_d)
        flatxcorr = xcorr.flatten(order='C')

        # The column can be used to form toeplitz matrix, making d^T d into D^T D
        column = np.zeros([autocorr.shape[0]])
        column[0:int(length/2)+1] = autocorr[int(length/2):]
        # toep = toeplitz(column)

        # regularisation
        padding = [0] * (len(trace_d) - 1) # n-1
        # D^T p + a d^T d z
        flatxcorr += stabilisation * column[0] * np.array(padding + [1] + padding)
        #D^T D + a d^t d I
        column[0] = (1 + stabilisation) * column[0]

        # solve for filter v
        v = scipy.linalg.solve_toeplitz(column, flatxcorr)

        tv = T * v / norm(v)

        # _, res_data = read_ttr_residual('res-csref00350-iter00001fwd1.ttr')
        # res_data = res_data[trace, :]
        # plt.figure()
        # plt.plot(res_data, label='residual file', alpha=0.7)
        # plt.plot(tv, label='T*v', alpha=0.7)
        # plt.plot(unfold_tv, label='unfold tv', alpha=0.7)
        # plt.legend()
        # plt.show()

        f = tv.T @ tv

        filters.append(tv) 
        f_values.append(f)


    # save filter to csv
    filters = np.array(filters)
    np.save(filename + "_filter_output.csv", filters)

    
    if verbose:
        trace = np.arange(filters.shape[0])
        plt.figure()
        plt.xlabel("$trace$")
        plt.ylabel("$time$")
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["blue","white","red"])
        extent = [trace[0]-0.5, trace[-1]+0.5, filters.shape[1] * time_interval * down_sampling, 0]
        plt.imshow(filters.T, cmap=cmap, interpolation='none', extent=extent, aspect="auto", norm=MidPointNorm(midpoint=0))
        plt.show()

    # return fnl1 value
    return sum(f_values)

# filename = 'compare-csref02398-iter00029fwd2.ttr'
# filename = 'compare-csref02398-iter00019fwd2.ttr'
filename = 'compare-csref00350-iter00001fwd1.ttr'
# filename='compare-csref00450-iter00025fwd1.ttr'
filter_plot(filename=filename, stabilisation=0.01, scale=0.05, wtype="exponential", wmin=True)

# f_values = []
# iterations = ["01", "10", "20", "30"]
# for iteration in iterations:
#     filename = 'compare-csref00350-iter000{}fwd1.ttr'.format(iteration)
#     f = filter_plot(filename=filename, stabilisation=0.01, scale=0.05, wtype="exponential", wmin=True)
#     print(filename, f)
#     f_values.append(f)


# plt.plot(list(map(int, iterations)), f_values)
# plt.show()


# T = apply_weight(299, 0.05, wtype='exponential', wmin=True)
# Tm = apply_weight(299, 0.05, wtype='exponential', wmin=False)
# plt.plot(T, label= "minimization")
# plt.plot(Tm, label= "maximization")


# for t in [0.05, 0.2, 0.5, 0.75]:
#     T = apply_weight(299, t, wtype='exponential', wmin=True)
#     plt.plot(T, label=t)