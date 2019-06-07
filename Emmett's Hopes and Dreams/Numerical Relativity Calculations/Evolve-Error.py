import numpy as np
import scipy
from scipy import special
from matplotlib import pyplot as plt
%matplotlib inline

def evolvetest(array, tstart, tfinal, stepper, dudtrhs, CF, info):

    dt = (tfinal - tstart) / np.ceil((tfinal - tstart) / (abs(info['xvalues'][1] - info['xvalues'][0]) * CF))
    nt = int((tfinal - tstart) / dt)
    print(dt)
    newarray = array.copy()
    for n in range(nt):
        newarray = stepper(newarray, dt, info, dudtrhs)
    return newarray

def error(array, tstart, tfinal, stepper, dudtrhs, CF, info):

    arrayanalytic = np.exp(-2 * np.cos(2 * np.pi * (info['xvalues'] - (tfinal - tstart))))
    arraynumeric = evolvetest(array, tstart, tfinal, stepper, dudtrhs, CF, info)
    return np.log10(abs(arrayanalytic - arraynumeric))
