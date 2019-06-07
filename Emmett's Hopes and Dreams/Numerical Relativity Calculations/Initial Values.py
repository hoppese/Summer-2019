import numpy as np
import scipy
from scipy import special
from matplotlib import pyplot as plt
%matplotlib inline

def x(xmin, xmax, nx):
    return np.linspace(xmin, xmax, nx, endpoint = True)

def u(xmin, xmax, nx):
    return np.exp(-2 * np.cos(2 * np.pi * x(xmin, xmax, nx)))

def xvalues(minx, maxx, nk, npe, r):

    xvalues = []
    for i in range(0, nk):
        for j in ((rtemp+1) * ((maxx - minx) / nk) / 2 + (i * (maxx - minx)) / nk):
            xvalues.append(j)

    xvalues = np.array(xvalues)

    return xvalues

def initial_u(info, equation):

    uvalues = equation(info['xvalues'])
    return uvalues
