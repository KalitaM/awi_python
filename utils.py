import struct
import numpy as np

def read_ttr_compare(filename):
    with open(filename, "rb") as f:
        f.read(4) # skip head
        n_comp = int.from_bytes(f.read(4), "little") # number of composite sources
        max_pr = int.from_bytes(f.read(4), "little") # max point receiver for each composite source
        timesteps = int.from_bytes(f.read(4), "little") # number of samples in each trace
        total_time = struct.unpack('f', f.read(4))[0]
        f.read(4) # skip tail
        skip_head = f.read(4)

        trace_nbytes = 4 * timesteps

        predict = []
        actual = []
        prev_ref = -1
        while skip_head:
            int.from_bytes(f.read(4), "little") # csref
            prref = int.from_bytes(f.read(4), "little")
            # samples = [struct.unpack('f', f.read(4))[0] for _ in range(timesteps)]
            samples = np.frombuffer(f.read(trace_nbytes), dtype="float32")
            if prref > prev_ref:
                predict.append(samples)
                prev_ref = prref
            else:
                actual.append(samples)
                break
            f.read(4) # skip tail
            skip_head = f.read(4)
        
        for _ in range(len(predict) - 1):
            f.read(16)
            # actual.append([struct.unpack('f', f.read(4))[0] for _ in range(timesteps)])
            actual.append(np.frombuffer(f.read(trace_nbytes), dtype="float32"))

        return total_time / timesteps, np.array(predict + actual)

def read_rcvrlist(filename):
    data = np.loadtxt(filename, dtype=np.int32, skiprows=1)
    return data

def read_ttr_residual(filename):
    with open(filename, "rb") as f:
        f.read(4) # skip head
        n_comp = int.from_bytes(f.read(4), "little") # number of composite sources
        max_pr = int.from_bytes(f.read(4), "little") # max point receiver for each composite source
        timesteps = int.from_bytes(f.read(4), "little") # number of samples in each trace
        total_time = struct.unpack('f', f.read(4))[0]
        f.read(4) # skip tail
        skip_head = f.read(4)

        trace_nbytes = 4 * timesteps


        predict = []
        prev_ref = -1
        while skip_head:
            int.from_bytes(f.read(4), "little") # csref
            prref = int.from_bytes(f.read(4), "little")
            # print(prref)
            # samples = [struct.unpack('f', f.read(4))[0] for _ in range(timesteps)]
            samples = np.frombuffer(f.read(trace_nbytes), dtype="float32")

            if prref > prev_ref:
                predict.append(samples)
                prev_ref = prref
            else:
                actual.append(samples)
                break
            f.read(4) # skip tail
            skip_head = f.read(4)
        
        return total_time / timesteps, np.array(predict)


from matplotlib.colors import Normalize
from numpy import ma
from matplotlib import cbook

class MidPointNorm(Normalize):    
    def __init__(self, midpoint=0, vmin=None, vmax=None, clip=False):
        Normalize.__init__(self,vmin, vmax, clip)
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        if clip is None:
            clip = self.clip

        result, is_scalar = self.process_value(value)

        self.autoscale_None(result)
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if not (vmin < midpoint < vmax):
            raise ValueError("midpoint must be between maxvalue and minvalue.")       
        elif vmin == vmax:
            result.fill(0) # Or should it be all masked? Or 0.5?
        elif vmin > vmax:
            raise ValueError("maxvalue must be bigger than minvalue")
        else:
            vmin = float(vmin)
            vmax = float(vmax)
            if clip:
                mask = ma.getmask(result)
                result = ma.array(np.clip(result.filled(vmax), vmin, vmax),
                                  mask=mask)

            # ma division is very slow; we can take a shortcut
            resdat = result.data

            #First scale to -1 to 1 range, than to from 0 to 1.
            resdat -= midpoint            
            resdat[resdat>0] /= abs(vmax - midpoint)            
            resdat[resdat<0] /= abs(vmin - midpoint)

            resdat /= 2.
            resdat += 0.5
            result = ma.array(resdat, mask=result.mask, copy=False)                

        if is_scalar:
            result = result[0]            
        return result

    def inverse(self, value):
        if not self.scaled():
            raise ValueError("Not invertible until scaled")
        vmin, vmax, midpoint = self.vmin, self.vmax, self.midpoint

        if cbook.iterable(value):
            val = ma.asarray(value)
            val = 2 * (val-0.5)  
            val[val>0]  *= abs(vmax - midpoint)
            val[val<0] *= abs(vmin - midpoint)
            val += midpoint
            return val
        else:
            val = 2 * (val - 0.5)
            if val < 0: 
                return  val*abs(vmin-midpoint) + midpoint
            else:
                return  val*abs(vmax-midpoint) + midpoint

def apply_weight(filter_len, scale, wtype, wmin):
    # wmin true: minimize, false: maximize
    # wtype: weight function type
    # scale: function width
    zerolag = filter_len // 2
    weights = [0] * filter_len
    iscale = min(zerolag, round(abs(scale) * zerolag))
    if iscale <= 1:
        iscale = round(max(1.0, 0.05 * zerolag))

    if wtype == 'gaussian':
        if wmin:
            weights[zerolag] = 0.0
            for i in range(1, zerolag+1):
                weights[zerolag+i] = weights[zerolag-i] = 1 - np.exp(-i**2 / iscale ** 2)
        else:
            weights[zerolag] = 1.0
            for i in range(1, zerolag+1):
                weights[zerolag+i] = weights[zerolag-i] = np.exp(-i**2 / iscale ** 2)
    elif wtype == 'exponential':
        if wmin:
            weights[zerolag] = 0.0
            for i in range(1, zerolag+1):
                weights[zerolag+i] = weights[zerolag-i] = 1 - np.exp(-i / iscale)
        else:
            weights[zerolag] = 1.0
            for i in range(1, zerolag+1):
                weights[zerolag+i] = weights[zerolag-i] = np.exp(-i / iscale)
    elif wtype == 'linear':
        if wmin:
            weights[zerolag] = 0.0
            for i in range(1, iscale):
                weights[zerolag+i] = weights[zerolag-i] = i / iscale
            for i in range(iscale, zerolag+1):
                weights[zerolag+i] = weights[zerolag-i] = 1.0
        else:
            weights[zerolag] = 1.0
            for i in range(1, iscale):
                weights[zerolag+i] = weights[zerolag-i] = 1 - i / iscale
            for i in range(iscale, zerolag+1):
                weights[zerolag+i] = weights[zerolag-i] = 0
    elif wtype == 'sine':
        if wmin:
            weights[zerolag] = 0.0
            for i in range(1, iscale):
                weights[zerolag+i] = weights[zerolag-i] = 0.5 * (1 - np.cos(i*np.pi/iscale))
            for i in range(iscale, zerolag+1):
                weights[zerolag+i] = weights[zerolag-i] = 1.0
        else:
            weights[zerolag] = 1.0
            for i in range(1, iscale):
                weights[zerolag+i] = weights[zerolag-i] = 0.5 * (1 + np.cos(i * np.pi/iscale))
            for i in range(iscale, zerolag+1):
                weights[zerolag+i] = weights[zerolag-i] = 0
    else:
        print("placeholder")
    return weights

# import matplotlib.pyplot as plt

# filter = 6141
# scale = 0.05

# plt.plot(apply_weight(filter, scale, "gaussian", True), label="gaussian")
# plt.plot(apply_weight(filter, scale, "linear", True), label="linear")
# plt.plot(apply_weight(filter, scale, "exponential", True), label="exponential")
# plt.plot(apply_weight(filter, scale, "sine", True), label="sine")

# plt.legend()


# plt.figure()
# plt.plot(apply_weight(filter, scale, "gaussian", False), label="gaussian")
# plt.plot(apply_weight(filter, scale, "linear", False), label="linear")
# plt.plot(apply_weight(filter, scale, "exponential", False), label="exponential")
# plt.plot(apply_weight(filter, scale, "sine", False), label="sine")

# plt.legend()

# plt.show()

