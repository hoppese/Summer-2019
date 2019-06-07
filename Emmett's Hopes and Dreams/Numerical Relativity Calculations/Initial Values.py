import numpy
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline

def x(xmin, xmax, nx):
    return numpy.linspace(xmin, xmax, nx, endpoint = False)

def u(xmin, xmax, nx):
    #return numpy.sin(2 * np.pi * x(xmin, xmax, nx))
    return numpy.exp(-2 * numpy.cos(2 * numpy.pi * x(xmin, xmax, nx)))

def xvalues(minx, maxx, nk, npe, rtemp):
    #r , Minv_ref, MinvS_ref = ReferenceElement(npe - 1)


    xvalues = []
    for i in range(0, nk):
        for j in ((rtemp+1) * ((maxx - minx) / nk) / 2 + (i * (maxx - minx)) / nk):
            xvalues.append(j)

    xvalues = np.array(xvalues)

    return xvalues

def initial_u(minx, maxx, nk, npe, rtemp, equation):

    #print(type(xvalues))
    #print(xvalues)
    uvalues = equation(xvalues(minx, maxx, nk, npe, rtemp))
    return uvalues
