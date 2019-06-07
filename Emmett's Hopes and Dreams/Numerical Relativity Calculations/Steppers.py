import numpy as np
import scipy
from scipy import special
from matplotlib import pyplot as plt
%matplotlib inline

def rk4step(array, dt, info, dudtrhs):
    w1 = dudtrhs(array, info)
    w2 = dudtrhs(array + 0.5 * dt * w1, info)
    w3 = dudtrhs(array + 0.5 * dt * w2, info)
    w4 = dudtrhs(array + dt * w3, info)

    return array + (dt / 6) * (w1 + 2 * w2 + 2 * w3 + w4)

def rk2step(array, dt, info, dudtrhs):
    w1 = dudtrhs(array, info)
    w2 = dudtrhs(array + 0.5 * dt * w1, info)
    return array + dt * w2

def fe_step(array, dt, info, dudtrhs):
    newarray = array + dt * dudtrhs(array, info)
    return newarray
