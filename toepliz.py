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
    for trace in range(1):
        trace = 100
    # for trace in range(p.shape[0]):
        # if trace % 15 == 0:
        #     print(trace)
        
        # # downsampling
        trace_p = list(p[trace, ::down_sampling])
        trace_d = list(d[trace, ::down_sampling])

        # toeplitz matrix
        padding = [0] * (len(trace_d) - 1) # n-1
        c, r = trace_d + padding + padding, [trace_d[0]] + padding + padding # 3n-2, 2n-1

        padding_p = np.array(padding + trace_p + padding) # 3n-2

        D = toeplitz(c,r)

        dtd = D.T @ D # 2n-1 * 3n-2 * 3n-2 * 2n-1

        dtp = D.T @ padding_p # 2n-1 * 3n-2 * 3n-2 * 1

        # identity filter
        nsamples = len(trace_p)
        filterI = np.zeros([nsamples],dtype=np.float32)
        filterI[int(nsamples/2)] = 1.0
        
        d_pad = scipy.signal.convolve(trace_d,filterI)
        p_pad = scipy.signal.convolve(trace_p,filterI)
        length = d_pad.shape[0]

        autocorr1 = scipy.signal.correlate(trace_d,trace_d)
        xcorr = scipy.signal.correlate(trace_p, trace_d)
        flatautocorr1 = autocorr1.flatten(order='C')
        flatxcorr = xcorr.flatten(order='C')
        lengthautocorr1 = autocorr1.shape[0]
        column = np.zeros([lengthautocorr1])
        column[0:int(length/2)+1] = autocorr1[int(length/2):]

        # regularisation
        column[0] = (1 + stabilisation) * column[0]
        dot_product = np.array(trace_d).T.dot(np.array(trace_d))
        dtp += stabilisation * dot_product * np.array(padding + [1] + padding)
        dtd += stabilisation * dot_product * np.eye(dtd.shape[0]) # stablizing by adding small diagonal
        flatxcorr += stabilisation * dot_product * np.array(padding + [1] + padding)

        dataw = scipy.linalg.solve_toeplitz(column,flatxcorr)
        # fold_dtd = toeplitz(column)
        # dataw = solve(fold_dtd,flatxcorr)

        # solve  (D^T D + a d^t d I) v = D^T p + a d^T d z;   z = [0..1..0] 2n-1, a = stablisation
        v = solve(dtd, dtp)

        plt.figure()
        plt.plot(v / norm(v), label='v', alpha=0.6)
        plt.plot(dataw.T / norm(dataw.T), label='unfold_v', alpha=0.6)
        plt.legend()
        # plt.show()

        tv = T * v / norm(v)

        fold_tv = T * dataw.T / norm(dataw.T)

        _, res_data = read_ttr_residual('res-csref00350-iter00001fwd1.ttr')

        res_data = res_data[trace, :]

        plt.figure()
        plt.plot(res_data / max(res_data[:-500]), label='residual file', alpha=0.7)
        plt.plot(tv / max(tv[:-500]), label='T*v', alpha=0.7)
        plt.plot(fold_tv / max(fold_tv[:-500]), label='fold tv', alpha=0.7)


        plt.legend()
        plt.show()

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